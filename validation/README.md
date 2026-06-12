# FoilCFD validation anchors

Reference datasets FoilCFD checks itself against. None of these are FoilCFD
output: they are wind-tunnel measurements and XFOIL panel-method polars for
the LS(1)-0413 / GA(W)-2 family, carried over (with full provenance) from the
Glasair III VG study at `L:\Dev\glasair_vg_sim`. They anchor the milestone
M3+ trend checks described at the bottom of this page.

## What lives where

```
validation/
  nasa/SOURCES.md          # NTRS provenance + digitization method/honesty notes
  nasa/digitized/*.csv     # hand-digitized NASA lift curves (alpha,cl)
  xfoil/polar_*.csv        # XFOIL 6.99 polars (7-column contract)
```

## NASA wind-tunnel anchors (`nasa/digitized/`)

Digitized by visual raster reads from image-only NTRS microfiche scans; the
full method, axis-calibration procedure, per-point uncertainty (typically
alpha +/- 0.2 deg, cl +/- 0.02), and honesty notes about what was NOT
digitized are in `nasa/SOURCES.md`.

| CSV | Source | Series |
|---|---|---|
| `nasa_tmx72843_fig5_Re2.2e6_trip.csv` | McGhee & Beasley, NASA TM X-72843 ([NTRS 19790004829](https://ntrs.nasa.gov/citations/19790004829)), fig 5 | LS(1)-0413, Re 2.2e6, M 0.15, transition fixed at x/c 0.075 |
| `nasa_tmx72843_fig5_Re4.3e6_trip.csv` | same report, fig 5 | LS(1)-0413, Re 4.3e6, tripped |
| `nasa_tmx72843_fig5_Re6.4e6_trip.csv` | same report, fig 5 | LS(1)-0413, Re 6.4e6, tripped |
| `nasa_wentz_cr145139_fig2a_Re2.2e6_trip.csv` | Wentz, NASA CR-145139 / WSU AR 76-2 ([NTRS 19790001850](https://ntrs.nasa.gov/citations/19790001850)), fig 2(a) | GA(W)-2 flap-nested, Re 2.2e6, M 0.13, strips 0.05c/0.10c |

All four are FIXED-transition data; TM X-72843 contains no free-transition
lift curve for the 0413 (see the honesty notes in `SOURCES.md`). The two
independent Re 2.2e6 sources agree on Clmax within 0.02 and stall AoA within
0.5 deg, which bounds the digitization error.

**CSV contract (NASA files):** `#`-prefixed provenance/uncertainty header
lines, then a literal `alpha,cl` header row, then comma-separated numeric
rows. Alpha in degrees, cl dimensionless. Filenames embed an `Re<value>`
token (e.g. `Re2.2e6`) so a reader can match a polar to a target Reynolds
number without parsing the comments.

## XFOIL panel-method anchors (`xfoil/`)

XFOIL 6.99 (Drela, MIT) viscous polars for the LS(1)-0413 section
(`airfoils/ls413.dat`), e^N transition with Ncrit = 9, at Re 1.5e6 / 3e6 /
6e6, each in free-transition (`_free`) and tripped (`_trip`, strips matching
the NASA tests) variants — six files, `polar_Re<x>M_N9_<free|trip>.csv`.
XFOIL is itself a model, not truth; it is included because it provides smooth
dense polars (including drag, which the NASA scans were too noisy to
digitize) and a second independent reference between the sparse experimental
Reynolds numbers.

**CSV contract (XFOIL files), 7 columns:**

```
alpha,cl,cd,cdp,cm,xtr_top,xtr_bot
```

`alpha` deg; `cl`/`cd`/`cdp` (pressure-drag part)/`cm` dimensionless;
`xtr_top`/`xtr_bot` are transition locations x/c on the upper/lower surface
(1.0 = no transition before the TE). Header row first, no comment lines.

## How FoilCFD milestone M3+ uses these

FoilCFD runs at a reduced *effective* Reynolds number (~1e4-1e5), so absolute
Cl/Cd cannot — and must not — be gated against these Re 1.5-6.4e6 anchors.
What M3+ automates instead is **trend agreement**, which is exactly what the
VG-placement mission needs:

1. **M3 lift-curve trend check (automated):** run the LS(1)-0413 (and NACA
   0012/2412) across an AoA sweep; assert that the lift-curve slope in the
   linear range is positive and of sane magnitude (anchors give ~0.11/deg),
   that cl(0 deg) has the correct sign/order for the cambered section
   (anchors: ~0.45-0.47), and that cl rises monotonically through ~8 deg and
   breaks/drops by 16+ deg, mirroring the digitized stall behavior. Compared
   quantity is the shape of cl(alpha), never the absolute level.
2. **Reynolds-trend sanity:** the three TM X-72843 curves show Clmax growing
   with Re (1.69 / 1.95 / 2.04). FoilCFD displays effective-vs-target Re and
   uses this anchor set to document the expected direction of the gap, so the
   README's honesty claims are backed by data in-repo.
3. **VG-delta framing (M4/M5):** VG-on vs VG-off deltas on identical grids
   are the product. The anchors establish the trustworthy clean-foil baseline
   shape that the deltas are measured against; the Wentz & Seetharam GA(W)-1
   VG findings cited in `docs/CITATIONS.md` give the qualitative expectation
   (separation suppressed, slope/Clmax up, stall reshaped).

When comparing at an experimental Reynolds number, use the matching-`Re`
file directly; do not interpolate between anchor curves — report a
no-matching-Re note instead (same policy the source project used).
