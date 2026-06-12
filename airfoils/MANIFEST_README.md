# Aircraft -> Airfoil Manifest

`aircraft_manifest.csv` maps popular aircraft (EAA/EAB homebuilts first, then
certified GA, warbirds, and gliders) to the airfoil section(s) their wings use,
so FoilCFD users can search for their aircraft and load the right geometry.

## File format

UTF-8 CSV with a header row. Fields containing commas must be double-quoted
(standard CSV escaping). Columns:

| column         | meaning                                                            |
|----------------|--------------------------------------------------------------------|
| `manufacturer` | Airframe maker (historical name as used by the source guide)       |
| `model`        | Aircraft model / variant range                                     |
| `category`     | One of `homebuilt`, `GA-single`, `GA-twin`, `trainer`, `aerobatic`, `LSA`, `warbird`, `glider` |
| `airfoil_root` | Root airfoil designation, verbatim from the source                 |
| `airfoil_tip`  | Tip airfoil designation (repeated if identical to root)            |
| `dat_root`     | Matching coordinate file in `airfoils/uiuc/` (or `airfoils/`), empty if none on disk |
| `dat_tip`      | Same, for the tip section                                          |
| `notes`        | Caveats: canard layouts, factory mods, year breaks, verification status |

Conventions used in this manifest:

- Biplanes (Pitts, etc.): the root column holds the upper-wing section and the
  tip column the lower-wing section, flagged in `notes`.
- Canard aircraft (VariEze, Long-EZ, Cozy, Velocity): the MAIN wing is listed,
  not the canard, flagged in `notes`.
- When the designation is "X mod" and only the unmodified base section exists
  on disk, the base `.dat` is linked and `notes` says so. Treat it as an
  approximation of the as-built wing.
- `dat_*` is left empty rather than guessed when no coordinate file exists.

## Provenance

Primary source: Dave Lednicer, "The Incomplete Guide to Airfoil Usage",
https://m-selig.ae.illinois.edu/ads/aircraft.html (retrieved 2026-06-11).
Airfoil designations are quoted verbatim from that guide unless `notes` says
otherwise. Entries not covered by the guide (e.g. RANS S-21, Sling) were
cross-checked against manufacturer documentation and aviation press and are
marked in `notes`; anything that could not be confirmed is tagged
`unverified`.

Coordinate files come from the UIUC Airfoil Coordinates Database
(https://m-selig.ae.illinois.edu/ads/coord_database.html), mirrored locally in
`airfoils/uiuc/`. `ls413.dat` also exists at `airfoils/ls413.dat`.

## Adding your own aircraft

Append a row to `aircraft_manifest.csv` following the column spec above:

1. Find your wing's section (type certificate data sheet, kit plans, or the
   Lednicer guide).
2. Use the designation exactly as documented - do not normalize it.
3. Point `dat_root`/`dat_tip` at a file that actually exists in
   `airfoils/uiuc/`; leave empty otherwise.
4. Keep `notes` under 100 characters and comma-free (or quote the field).

## Disclaimer

Manufacturers frequently modify published sections (drooped leading edges,
thickness changes, fairing blends) without renaming them, and kit designers
sometimes keep their exact coordinates proprietary. The listed section may
differ from the metal/glass on your aircraft. Verify against your type
certificate data sheet or kit documentation before trusting any analysis
based on this manifest.
