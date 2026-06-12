# NASA validation source documents

Experimental anchors for the Phase-1 clean-section validation gate and the
Phase-3 deflected-aileron comparisons. Retrieved 2026-06-10 from the NASA
Technical Reports Server (`https://ntrs.nasa.gov/api/citations/<id>/downloads/<id>.pdf`).

| File | NTRS id | Contents / why we need it |
|---|---|---|
| `NASA_TMX72843_thickness_effects_GAW_family.pdf` | 19790004829 | McGhee & Beasley, *Effects of thickness on the aerodynamic characteristics of an initial low-speed family of airfoils for general aviation applications* (TM X-72843). Cl/Cd/Cm vs alpha across Re for the GA(W) family incl. the 13% GA(W)-2 = LS(1)-0413 — primary clean-section gate data (Clmax within 10%, stall AoA within 2 deg). |
| `NASA_GAW2_aileron_flap_spoiler.pdf` | 19790001850 | Wentz, *Wind tunnel tests of the GA(W)-2 airfoil with 20% aileron, 25% slotted flap, 30% Fowler flap and 10% slot-lip spoiler* (NASA CR-145139 / Wichita State AR 76-2). The Glasair III aileron measures 19.87% chord on this exact section — near-direct experimental anchor for the deflected-aileron 2.5D cases in Study 1. |
| `NASA_modified_13pct_airfoil.pdf` | 19790016789 | McGhee & Beasley, *Low-speed wind tunnel results for a modified 13-percent-thick airfoil*. Secondary reference; useful if the as-built wing turns out to deviate from the published section (VT slides cite t/c = 12.5%). |

Index of all LS(1)-0413 literature: https://m-selig.ae.illinois.edu/ads/ref/ls0413_ref.html

## Figure / page map (1-based PDF pages)

### TM X-72843 (54 pages)

| PDF page | Figure | Contents |
|---|---|---|
| 27 | 5 | **Section characteristics for LS(1)-0413, M = 0.15, transition fixed at x/c = 0.075; R = 2.2 / 4.3 / 6.4 x 1e6 (circle / square / diamond). The Phase-1 Clmax/stall gate source — digitized below.** |
| 28 | 6 | Same layout for LS(1)-0417 (not our section). |
| 29-33 | 7(a)-(e) | Roughness off/on across R = 2.0-9.0e6 for **LS(1)-0421 only** — the report has no transition-free lift curve for the 0413, so no free-transition CSV exists for it (see honesty notes). |
| 34 | 8 | Mach sweep, 0413. |
| 35-40 | 9 | Chordwise pressure distributions, 0413, with tabulated alpha/cl legends. |
| 41-44 | 10-11 | Spanwise drag / thickness-effect summaries. |
| 45 | 12 | Clmax vs Re per airfoil (0413 = circles) — used as the digitization cross-check. |
| 51-53 | 17(a)-(c) | Experiment vs theory per airfoil. |

### CR-145139 (82 pages)

| PDF page | Figure | Contents |
|---|---|---|
| 25 | 2(a) | **Basic GA(W)-2 lift curve, flap nested, R = 2.2e6, M = 0.13, transition strips at 0.05c upper / 0.10c lower: WSU balance data (circles) + NASA Langley reference curve (dashed). WSU circles digitized below.** |
| 25-26 | 2(b),(c) | Moment and drag panels of the same comparison. |
| 27-33 | 3 | 20% aileron deflection series (Phase-3 anchor, digitization deferred to M3+). |
| 34-48 | 4-9 | Slotted/Fowler flap optimization and effectiveness. |

## Digitization (M1, 2026-06-10)

Digitized lift-curve CSVs live in `digitized/` and follow the
`validation/compare_gate.py` reader contract: `#` provenance headers, an
`alpha,cl` header row, comma-separated numeric rows, and an `Re<value>` token
in the filename that `find_nasa_polar` matches at 1% tolerance.

| CSV | Source | Series | Points |
|---|---|---|---|
| `nasa_tmx72843_fig5_Re2.2e6_trip.csv` | TM X-72843 fig 5, p27 | circles, R 2.2e6, trip 0.075c | 21 |
| `nasa_tmx72843_fig5_Re4.3e6_trip.csv` | TM X-72843 fig 5, p27 | squares, R 4.3e6, trip 0.075c | 22 |
| `nasa_tmx72843_fig5_Re6.4e6_trip.csv` | TM X-72843 fig 5, p27 | diamonds, R 6.4e6, trip 0.075c | 22 |
| `nasa_wentz_cr145139_fig2a_Re2.2e6_trip.csv` | CR-145139 fig 2(a), p25 | WSU circles, R 2.2e6, strips 0.05c/0.10c | 15 |

Method (the scans are image-only microfiche copies; everything is a visual
read, nothing vector-extracted):

1. Pages rendered with `scripts/digitize_helper.py` (PyMuPDF) at 300 dpi for
   calibration and 600 dpi (TM X-72843) / 300 dpi (CR-145139) for the reads.
2. Axis calibration by darkness-profiling the long gridlines and anchoring
   the line lattice on the printed tick labels (plus the one OCR-locatable
   `a = 8 deg` label on TM p27); fig 5 lattice: 0.2 cl / 4 deg majors with
   0.04 cl / 0.4 deg minors; fig 2(a): 0.2 cl / 2 deg lines.
3. Calibration verified closed-loop by overlaying the predicted lattice on
   the scan and checking every labeled tick (procedure kept in the gitignored
   `results/digitize_scratch/` overlays; regenerate with the helper script).
4. Symbol centers read visually from zoomed calibrated tiles and converted
   through the lattice; each CSV header records the per-file pixel resolution
   and the resulting uncertainty (alpha +/- 0.2 deg, cl +/- 0.02 typical).

Cross-checks recorded in the CSV headers: cl(0) = 0.45-0.47 (expected
0.4-0.5 for this section), dcl/dalpha over [-2,+6] deg = 0.107-0.119 per deg
(expected ~0.11), the two independent R = 2.2e6 tripped sources agree on
Clmax within 0.02 and stall AoA within 0.5 deg, and the fig 5 Clmax-vs-Re
trend (1.69 / 1.95 / 2.04) tracks the fig 12 summary curve for the 0413.

Honesty notes — what was NOT digitized and why:

* TM X-72843 fig 5 below alpha ~2 deg: the three Reynolds series plot within
  one symbol width, so a shared cluster read (with widened uncertainty,
  noted per file) is recorded for all three Re; a faint overlapped cluster
  near alpha = -0.6 deg could not be separated per-Re and was omitted.
* TM X-72843 carries **no transition-free 0413 lift curve** (fig 7's
  roughness-off series covers the 0421 only), so all digitized 0413 data are
  fixed-transition; free-transition RANS points have no NASA anchor yet.
* CR-145139 fig 2(a): only the WSU circle symbols were digitized; the dashed
  NASA Langley line is a faired curve without test points. The figure's
  drag panel (fig 2(c)) is too noisy at 300 dpi for honest cd reads.
* The experiment Reynolds numbers (2.2 / 4.3 / 6.4e6) do not coincide with
  the default validation matrix points (1.5 / 3 / 6e6); the gate reports an
  explicit no-matching-Re note rather than interpolating between figures.
  Run the gate at `--re 2.2e6` (etc.) to compare against these anchors.
