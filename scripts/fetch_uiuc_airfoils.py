#!/usr/bin/env python3
# fetch_uiuc_airfoils.py - Download the full UIUC airfoil coordinate database
# into airfoils/uiuc/ for FoilCFD's .dat loader. Helper script, NOT runtime.
# FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#
# Behavior (per FOILCFD_PLAN.md section 15):
#   * Scrapes https://m-selig.ae.illinois.edu/ads/coord_database.html for every
#     link matching coord/*.dat (~1600 files, ~15 MB total).
#   * Downloads into <repo>/airfoils/uiuc/ with a polite ~75 ms delay between
#     requests so we are a good citizen toward the UIUC server.
#   * Idempotent / resumable: files that already exist locally (non-empty) are
#     skipped, so re-running after an interruption picks up where it left off.
#   * Transient failures (HTTP 5xx, timeouts, connection resets) are retried
#     twice with a short backoff before the file is recorded as failed.
#   * Python stdlib only (urllib + html.parser); no third-party packages.
#   * Windows-console safe: ASCII-only output, stdout wrapped with
#     errors='replace' so odd airfoil names can never crash the script.

import io
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

# ----------------------------------------------------------------------------
# Console safety: legacy Windows consoles may use a non-UTF-8 code page, and a
# stray non-ASCII character in a filename must never kill a 1600-file run.
# Re-wrap stdout/stderr so any unencodable character degrades to '?' instead.
# ----------------------------------------------------------------------------
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name)
    if hasattr(_stream, "buffer"):
        setattr(sys, _stream_name, io.TextIOWrapper(
            _stream.buffer, encoding="utf-8", errors="replace",
            line_buffering=True))

# Index page that links every coordinate file in the database.
INDEX_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
# Per-request courtesy delay (seconds). ~75 ms keeps the whole run at a few
# minutes while staying far below anything resembling a hammering rate.
REQUEST_DELAY_S = 0.075
# Retries for transient failures: 1 initial attempt + 2 retries = 3 tries.
MAX_TRIES = 3
RETRY_BACKOFF_S = 2.0
# Network timeout per request; the .dat files are tiny (a few KB each).
TIMEOUT_S = 30
# Identify ourselves honestly; some servers reject the default Python UA.
USER_AGENT = "FoilCFD-airfoil-fetch/1.0 (offline coordinate cache builder)"

# Destination directory, resolved relative to this script so the tool works
# regardless of the caller's current working directory: <repo>/airfoils/uiuc.
DEST_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "airfoils", "uiuc"))


class CoordLinkParser(HTMLParser):
    """Collect every <a href> on the index page that points at coord/*.dat.

    The UIUC page also links coord_updates/, ref/ pages, images, etc.; we
    keep strictly the coord/<name>.dat pattern the plan asks for. Order is
    preserved and duplicates (the page links some files more than once) are
    removed via the seen-set.
    """

    def __init__(self):
        super().__init__()
        self.links = []        # ordered unique list of href strings
        self._seen = set()     # dedupe guard

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() != "href" or not value:
                continue
            # Normalize: strip any query/fragment, collapse ./ prefixes.
            href = value.split("#", 1)[0].split("?", 1)[0].strip()
            while href.startswith("./"):
                href = href[2:]
            # Keep only relative links of the exact form coord/<file>.dat
            # (case-insensitive extension; the database is all lowercase but
            # we stay defensive about future edits to the page).
            if href.lower().startswith("coord/") and \
                    href.lower().endswith(".dat") and \
                    href.count("/") == 1:
                if href not in self._seen:
                    self._seen.add(href)
                    self.links.append(href)


def fetch_url(url):
    """GET a URL and return raw bytes, retrying transient failures.

    Transient = HTTP 5xx, 429 (rate limited), URL/socket-level errors
    (timeouts, resets). Permanent errors (404 etc.) raise immediately so a
    bad link is reported once instead of retried pointlessly.
    """
    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            # 4xx other than 429 will not heal with a retry; fail fast.
            if err.code < 500 and err.code != 429:
                raise
            last_err = err
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
            last_err = err
        # Back off briefly before the next try (skipped after the last one).
        if attempt < MAX_TRIES:
            time.sleep(RETRY_BACKOFF_S * attempt)
    raise last_err


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: pull the index page and extract every coord/*.dat link.
    # ------------------------------------------------------------------
    print("Fetching index: %s" % INDEX_URL)
    try:
        index_bytes = fetch_url(INDEX_URL)
    except Exception as err:
        print("ERROR: could not fetch the index page: %s" % err)
        return 1
    # The page declares no exotic encoding; decode permissively.
    parser = CoordLinkParser()
    parser.feed(index_bytes.decode("utf-8", errors="replace"))
    links = parser.links
    if not links:
        print("ERROR: index parsed but no coord/*.dat links found "
              "(page layout may have changed).")
        return 1
    print("Index lists %d coordinate files." % len(links))

    # ------------------------------------------------------------------
    # Step 2: download each file, skipping anything already on disk so the
    # run is resumable. Progress is printed every 100 files to keep the
    # console readable over ~1600 downloads.
    # ------------------------------------------------------------------
    base = INDEX_URL.rsplit("/", 1)[0] + "/"   # .../ads/
    downloaded = 0
    skipped = 0
    failed = []
    for i, href in enumerate(links, 1):
        fname = href.split("/", 1)[1]
        # Guard against any path trickery in a scraped link.
        fname = os.path.basename(fname)
        dest = os.path.join(DEST_DIR, fname)
        # Skip files that already exist and are non-empty (idempotent rerun);
        # a zero-byte file is treated as a previous failed write and redone.
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            skipped += 1
            continue
        url = urllib.parse.urljoin(base, urllib.parse.quote(href))
        try:
            data = fetch_url(url)
        except Exception as err:
            failed.append((fname, str(err)))
            print("  FAIL %s: %s" % (fname, err))
            continue
        # Write atomically-ish: temp file then rename, so an interrupted run
        # never leaves a truncated .dat that a later rerun would skip.
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
        downloaded += 1
        if (downloaded % 100) == 0:
            print("  ... %d downloaded (%d/%d processed)"
                  % (downloaded, i, len(links)))
        # Courtesy delay between requests (only when we actually hit the
        # network -- skipped files cost the server nothing).
        time.sleep(REQUEST_DELAY_S)

    # ------------------------------------------------------------------
    # Step 3: one summary line the caller (or CI log reader) can grep for.
    # ------------------------------------------------------------------
    total_on_disk = len([f for f in os.listdir(DEST_DIR)
                         if f.lower().endswith(".dat")])
    print("SUMMARY: %d downloaded, %d skipped (already present), %d failed; "
          "%d .dat files now in %s"
          % (downloaded, skipped, len(failed), total_on_disk, DEST_DIR))
    if failed:
        print("Failed files (rerun the script to retry): "
              + ", ".join(name for name, _ in failed))
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
