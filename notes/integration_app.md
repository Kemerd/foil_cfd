# Integration notes — app shell (main.cpp, app/ui, app/camera, platform/)

## src/sim/lbm_core.cuh : CellFlag (z-face free-slip values)
Proposed:
    /// @brief Per-cell type flag. Values are stored in snapshots' flag-hash
    /// computation, so the numeric assignments are frozen alongside the lattice.
    enum class CellFlag : std::uint8_t {
        Fluid      = 0,
        Solid      = 1,
        Inlet      = 2,
        Outlet     = 3,
        SlipTop    = 4,
        SlipBottom = 5,
        SlipFront  = 6, ///< z=0 plane: free-slip wall (STL mode, plan 7.4).
        SlipBack   = 7, ///< z=nz-1 plane: free-slip wall (STL mode, plan 7.4).
    };
Reason: Plan 7.4 requires a UI toggle that switches the spanwise boundary from
periodic to free-slip WALLS for full-3D STL imports. The flag enum only covers
y-plane slip today. The app shell already stamps the raw byte values 6/7 onto
the z faces when the user picks "free-slip side walls" in the STL import modal
(see kCellFlagSlipFront/kCellFlagSlipBack constants in src/main.cpp), so the
stream-collide kernel must treat 6/7 as specular reflection in z exactly like
SlipTop/SlipBottom do in y.
Status: ACCEPTED + IMPLEMENTED (integration pass, 2026-06-11). CellFlag gained
SlipFront=6/SlipBack=7, lbm_core.cuh gained the kMirZ[19] mirror table (hand
verified against the frozen velocity table), and the fused kernel's pull loop
selects mirZ(q) at (x-cx, y-cy, z) for slip-z neighbors — symmetric with the
y-slip path. main.cpp now uses the enum values instead of raw bytes. Ghost
planes are never read in slip-wall mode (fluid z stays in [1,nz-2]).

## src/render/viz.h : Visualizer::uploadStlMesh
Proposed:
    /// @brief Upload (or replace) an imported STL render mesh, replacing the
    /// extruded-foil prism geometry while STL mode is active. Vertices are in
    /// lattice-cell coordinates (post applyNormalization). Pass an empty mesh
    /// to fall back to the airfoil prism from uploadGeometry().
    /// @param mesh Normalized mesh from the import flow (plan 7.2).
    void uploadStlMesh(const StlMesh& mesh);
Reason: Visualizer::uploadGeometry only accepts an AirfoilGeometry outline, so
an imported STL has no render-mesh path and the solid is invisible (particles
still trace it). The app shell currently sets VizSettings::showFoilMesh = false
on STL import and surfaces a status note; once this lands, main.cpp's
applyStlImport() will call it right after voxelization (call site marked with
a comment referencing this entry).
Status: ACCEPTED + IMPLEMENTED (integration pass, 2026-06-11). Implemented in
viz.cpp reusing the existing mesh VBO/shader (flat-shaded winding normals, file
normals only as degenerate fallback; empty mesh clears, uploadGeometry()
restores the prism). main.cpp's applyStlImport() now calls it and keeps
showFoilMesh enabled. StlMesh is forward-declared in viz.h to keep the header
light.

## src/sim/lbm_solver.h : LBMSolver::updateScaling (optional improvement)
Proposed:
    /// @brief Adopt a new LatticeScaling without reallocating or destroying
    /// the flow field (airspeed changes, plan 13: "changing airspeed does NOT
    /// invalidate — rescale via units, restart ramp"). Restarts the viscosity
    /// ramp and force-EMA gate; keeps f, flags, and the step pacer state.
    /// Grid dims and u_lat must match the init-time values.
    void updateScaling(const LatticeScaling& scaling);
Reason: Airspeed edits only move tau/dt; the lattice field is reusable as-is.
Lacking this, the app shell implements airspeed changes as shutdown() +
init(new scaling) + VRAM-snapshot restore (correct, but reallocates ~4 GB and
needs a captured snapshot to stay warm). If accepted, main.cpp's
handleAirspeedChanged() collapses to a single call — the workaround is marked
with a comment referencing this entry. LOW priority: the workaround works.
Status: DEFERRED (integration pass, 2026-06-11). The shutdown+init+restore
workaround in main.cpp is correct per plan §13 and exercised today; an
in-place rescale touches the solver's ramp/EMA/pacer internals for a purely
transient ~4 GB realloc win. Revisit alongside the v1.x FP16-storage work,
where init cost grows and the payoff doubles.
