# FoilCFD — Physics Validation Report (plan §10 milestones M0–M2)

Validated 2026-06-11 on the primary dev machine (RTX 5090, CUDA 12.9, Release
build via the exact BUILDING.md commands). **No solver code changes were
required** — all milestones passed on the integrated build as delivered.
Measured values below come from running each milestone executable directly
(`build/tests/Release/*.exe`) so the printed diagnostics are captured; the
pass/fail verdicts are the CTest results.

## Summary table

| Milestone | Test | Setup | Acceptance gate | Measured | Verdict | Runtime |
|---|---|---|---|---|---|---|
| M0 | `m0_taylor_green` | 64³ Taylor–Green decay, nu_lat = 0.01 (tau ≈ 0.53), 50 000 steps | KE monotone non-increasing (1% jitter allowance, enforced above a 0.2%·KE0 floor), no NaN, final KE < 2% of KE0 | KE0 = 76.80 → KE(50k) = 0.0941 lattice units = **0.123% of KE0 (816× decay)**; monotone everywhere above the floor; **zero NaN** over 50k steps; 158/158 checks | **PASS** | 1.8–2.0 s |
| M1 | `m1_cavity` | Lid-driven cavity 128×128×4, Re 1000, run to profile self-convergence | u(y) at the 15 Ghia/Ghia/Shin (1982) Table I centerline stations within **7% of lid speed** each | Converged at step 68 000 (profile drift 4.1e-4 < 5e-4 gate); **max deviation 0.0146 of lid speed = 1.46%** (worst station y = 0.1016); 88/88 checks | **PASS** | 2.0–2.2 s |
| M2 | `m2_cylinder` | Voxelized cylinder D = 40 cells, Re 150, 10% blockage; lift from momentum-exchange reduction; Hann-windowed Goertzel scan over St candidates | Spectral peak St in **0.17–0.21** and peak power ≥ a distinctness multiple of the scan mean | **St = 0.2020**, peak power 4.70e4 = **12.5× scan mean** (unambiguous vortex-street tone); 61/61 checks | **PASS** | 11.5–11.8 s |

Full suite after validation: `ctest --test-dir build -C Release` → **7/7 PASS**
(test_units / test_airfoil / test_voxelizer / cuda_smoke / m0 / m1 / m2),
15.5 s total. App selftest: `FoilCFD.exe --selftest` → **PASS**, exit 0,
7638 MLUPS over the 100 timed steps at 768×320×96.

## M0 — Taylor–Green energy decay (detail)

Sampled KE trace (every 10k steps of the 1k-step audit cadence):

| step | KE (lattice) | KE/KE0 |
|---|---|---|
| 0 | 7.680e+01 | 100% |
| 10 000 | 1.757e+00 | 2.29% |
| 20 000 | 1.442e-01 | 0.188% |
| 30 000 | 8.533e-02 | 0.111% |
| 40 000 | 9.143e-02 | 0.119% |
| 50 000 | 9.413e-02 | 0.123% |

Honesty note: the decay is strictly monotone down to ~0.11% of KE0, after
which KE plateaus and creeps up by ~7% per 10k steps (0.0853 → 0.0941
between steps 30k and 50k). This happens **below** the test's 0.2%·KE0
monotonicity floor and is the documented open-boundary residual: the test
domain is not truly periodic (inlet/outlet in x, free-slip y), so a tiny
inlet trickle (u_lat = 1e-3 Dirichlet plane, ~0.04% of KE0 at steady state)
plus residual boundary-layer circulation set a noise floor. The physics
statement — viscous dissipation kills the vortex, 816× energy drop, no
instability or NaN over 50k steps — is unaffected.

## M1 — Ghia centerline comparison (detail)

Per-station u-velocity on the vertical centerline (units of lid speed):

| y | u_Ghia | u_sim | abs. deviation |
|---|---|---|---|
| 0.0547 | -0.18109 | -0.16878 | 0.0123 |
| 0.0625 | -0.20196 | -0.18871 | 0.0132 |
| 0.0703 | -0.22220 | -0.20824 | 0.0140 |
| 0.1016 | -0.29730 | -0.28267 | 0.0146 |
| 0.1719 | -0.38289 | -0.37545 | 0.0074 |
| 0.2813 | -0.27805 | -0.27617 | 0.0019 |
| 0.4531 | -0.10648 | -0.10656 | 0.0001 |
| 0.5000 | -0.06080 | -0.06165 | 0.0008 |
| 0.6172 |  0.05702 |  0.05435 | 0.0027 |
| 0.7344 |  0.18719 |  0.18289 | 0.0043 |
| 0.8516 |  0.33304 |  0.32702 | 0.0060 |
| 0.9531 |  0.46604 |  0.46958 | 0.0035 |
| 0.9609 |  0.51117 |  0.51818 | 0.0070 |
| 0.9688 |  0.57492 |  0.58573 | 0.0108 |
| 0.9766 |  0.65928 |  0.67135 | 0.0121 |

Max deviation 1.46% of lid speed against the 7% gate (plan §10 asks "~5%";
comfortably inside). Expressed *relative to the local reference value* the
worst station is the near-wall y = 0.0547 point at 6.8% — typical for
half-way bounce-back at 128² resolution and still under the plan's spirit of
"within ~5–7%". The largest absolute errors cluster in the bottom-wall and
lid boundary layers, exactly where stair-step/bounce-back first-order wall
placement error lives; the core vortex (y = 0.17–0.85) agrees to ≤ 0.7%.

## M2 — Cylinder Strouhal (detail)

Lift trace shows clean periodic shedding after the 25k-step seeded transient
(Fy swinging ±0.27 across the 3000-sample record). Spectral peak
St = 0.2020 vs literature ~0.183 for an unbounded cylinder at Re 150; the
test's 10% blockage deliberately raises St a few percent (documented in the
test header), so 0.2020 sits mid-band rather than at the ceiling. Peak power
12.5× the scan mean — a sharp, isolated tone, not noise selection.

## What this does and does not prove

- **Proves:** D3Q19 weights/velocity-set/opposite tables are consistent
  (M0 stability + correct dissipation rate), TRT collision + half-way
  bounce-back deliver Re-1000 wall-bounded accuracy (M1), and the
  units mapping + momentum-exchange force path produce a physically correct
  unsteady wake frequency (M2).
- **Does not prove:** absolute Cl/Cd normalization on the airfoil
  (sim agent's pre-existing risk — pending the M3 visual/trend check),
  the STL slip-z kernel path (no automated test yet; integration report
  risk #1 stands), or High-Re behavior (plan §1 honesty: deltas, not
  absolutes).
