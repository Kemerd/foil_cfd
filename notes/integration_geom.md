# Integration notes — geom (airfoil + VG + voxelizer)

## No foreign-interface changes required

The geom module implemented against the existing headers as scaffolded; the
only header additions were to files this module owns (`src/geom/vg.h`).

## Facts downstream agents should know (informational, already true in tree)

### src/geom/vg.h : recommendedSweetSpotHeightBand (NEW, owned here)
    GuidanceBand recommendedSweetSpotHeightBand(float delta99_c);
plus constants `kLinSweetSpotMinDelta99 = 0.2f` / `kLinSweetSpotMaxDelta99 = 0.5f`.
Reason: the spec's Lin-2002 guidance includes the low-profile sweet spot
(h = 0.2-0.5 delta99) inside the full 0.1-1.0 band; the UI overlay can draw it
as the highlighted inner band. UI agent: call alongside recommendedHeightBand.
Status: INFORMATIONAL (header owned by geom)

### Catalog display names use ASCII " - ", not an em dash
`scanAirfoilDirectory` builds `"<filename> - <name line>"` with a plain ASCII
hyphen separator (the airfoil.h doc comment showed an em dash). Chosen because
ImGui's default font atlas does not cover U+2014; no caller binds to the
separator. Files whose first content line is numeric (headerless) list the
filename alone.

### Loader tolerances actually implemented
Selig + Lednicer auto-detect per plan 5.2 (line-2 values > 1, plus a
near-integer requirement that protects millimeter-scaled Selig files), CRLF,
tabs, comma separators, stray blank lines, duplicate LE points, `#`-prefixed
comment lines (our annotated `airfoils/ls413.dat` carries such a header),
lower-surface-first loops (re-canonicalized), and trailing annotation columns.
Verified: all 233 files under `airfoils/` (incl. `uiuc/`) load; rejection
reasons are human-readable with 1-based line numbers.

### VG voxelization conventions (consumers: render mesh, warm-start flow)
- `voxelizeVG`/`buildFlagsWithVGs` only ever convert `Fluid -> Solid`; domain
  face flags are never overwritten; z wraps periodically (spanwise-periodic BC).
- Vane roots are embedded 1.5 cells below the surface and re-follow the local
  surface frame along the vane length, so curvature/AoA can never open a gap
  under a vane. Devices are stamped analytically (0.45-cell sub-sampling) with
  a 1-cell minimum thickness — thin vanes cannot alias away.
- CounterRotatingPair orientation: `commonFlowDown == true` toes the trailing
  edges IN (the -z vane of a pair gets +beta). Units center on mid-span,
  spaced `pitch_c` apart.
- `buildFlagsWithVGs` runs `closeTrailingEdgeGaps` once at the end; callers
  must NOT run it again.

Status: INFORMATIONAL
