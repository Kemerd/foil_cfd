# FoilCFD performance record

## Selftest MLUPS (integration pass, 2026-06-11)

Measured by `FoilCFD.exe --selftest` (wall-clock over 100 fused steps after a
100-step warm-up, sync-bracketed; print added during integration):

| Machine | GPU | Grid | Throughput |
|---|---|---|---|
| Primary dev box (Win 11, VS 17.14, CUDA 12.9) | RTX 5090 (sm_120, driver 596.36) | 768 x 320 x 96 (23.6 M cells) | **8151 MLUPS (8.15 GLUPS)** — 0.289 s / 100 steps |

Assessment against plan section 11: the bandwidth ceiling estimate is ~10
GLUPS (152 B/cell at ~1.5 TB/s effective) and "5-7 is a good real number";
8.15 GLUPS clears the 2.0 GLUPS sanity floor by 4x and sits at the top of the
expected band, so the SoA pull-scheme streaming is coalescing correctly — no
optimization work warranted.

Context for the number: 8.15 GLUPS = ~345 steps/s at the default grid, inside
the plan 4.6 expectation of 200-380 steps/s. Test-suite runtimes on the same
card: m0 Taylor-Green 1.7 s, m1 cavity ~2 s, m2 cylinder ~12 s.

Note: the selftest measurement includes the per-step ghost-plane refresh
kernel and the NaN-watchdog cadence (every 200 steps), i.e. it is the honest
end-to-end step cost, not a bare-kernel microbenchmark. Macroscopic stores
(writeMacro) run only on the final step of each stepN batch (1 of the 100
timed steps), matching the plan 11 guidance that the 16 B/cell store is paid
only when a freshly-rendered field needs it.
