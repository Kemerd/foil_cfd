// Orbit camera (plan section 9): spherical-coordinate orbit around a target
// point with mouse-drag rotate, pan, and scroll zoom. All user input edits
// GOAL values; update() advances the displayed state toward the goals with
// critically damped springs so the view glides instead of snapping. Header
// keeps only state + accessors; the .cpp implements the math so main.cpp and
// the renderer stay free of it.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include "../core/vec.h"

namespace foilcfd {

/// @brief Orbit camera: position derived from (target, yaw, pitch, distance).
/// Angles in radians; pitch is clamped short of the poles so the view matrix
/// never degenerates. Distances/targets are in lattice-cell world units (the
/// renderer draws the domain in cell coordinates).
///
/// Smoothing model: every parameter exists twice — the smoothed value the
/// matrices are built from, and the goal the input handlers write. update()
/// integrates a critically damped spring per parameter (no overshoot, no
/// oscillation — the fastest non-ringing response for a given stiffness).
class OrbitCamera {
public:
    /// @brief Frame a domain of the given dimensions: target at the domain
    /// center, distance chosen so the whole grid fits the default FOV.
    /// Snaps (no animation) — used at init and on grid rebuilds.
    /// @param nx Domain x extent in cells.
    /// @param ny Domain y extent in cells.
    /// @param nz Domain z extent in cells.
    void frameDomain(int nx, int ny, int nz);

    /// @brief Frame an arbitrary region (e.g. the foil itself rather than the
    /// whole wind tunnel): target at @p center, distance fitting @p radius.
    /// Animated — the spring carries the view there. Clip planes keep tracking
    /// the full scene radius from the last frameDomain() call.
    /// @param center Region center in cell coordinates.
    /// @param radius Bounding-sphere radius of the region in cells.
    void frameRegion(const Vec3f& center, float radius);

    /// @brief Mouse-drag orbit: yaw/pitch deltas from pixel deltas.
    /// @param dxPixels Horizontal mouse movement since last frame.
    /// @param dyPixels Vertical mouse movement since last frame.
    void rotate(float dxPixels, float dyPixels);

    /// @brief Mouse-drag pan: moves the target in the camera's screen plane,
    /// scaled by distance so panning feels constant-speed at any zoom.
    void pan(float dxPixels, float dyPixels);

    /// @brief Scroll zoom: exponential distance scaling (each notch is a
    /// fixed percentage), clamped to sane near/far limits.
    /// @param scrollNotches Positive = zoom in.
    void zoom(float scrollNotches);

    /// @brief Advance the smoothed state toward the goals by @p dtSeconds
    /// using critically damped springs. Call once per frame before building
    /// matrices; large dt values are clamped so a hitch never destabilizes
    /// the integrator.
    void update(float dtSeconds);

    /// @brief Jump the smoothed state to the goals instantly (init, selftest,
    /// and "reset view" want zero animation).
    void snapToGoals();

    /// @brief Camera position in world (cell) space, derived from the
    /// SMOOTHED orbit parameters.
    Vec3f eye() const;

    /// @brief Current smoothed orbit target (center of interest).
    Vec3f target() const { return target_; }

    /// @brief View matrix for the current smoothed orbit state.
    Mat4f viewMatrix() const;

    /// @brief Perspective projection for the given viewport aspect; near/far
    /// planes track the framed domain size.
    /// @param aspect Viewport width / height.
    Mat4f projMatrix(float aspect) const;

    // ------ direct parameter access (UI "reset view", serialization) ------
    // Getters report the GOAL state (what the user asked for); setters snap
    // both goal and smoothed value so programmatic placement is immediate.

    float yaw()      const { return yawGoal_; }
    float pitch()    const { return pitchGoal_; }
    float distance() const { return distGoal_; }
    void  setYaw(float v);
    void  setPitch(float v);   // clamps to (-pi/2, pi/2) minus epsilon
    void  setDistance(float v);// clamps to [near limit, far limit]
    void  setTarget(const Vec3f& t);

private:
    // -- smoothed (displayed) state: matrices are built from these --
    Vec3f target_{0, 0, 0};   ///< Orbit center, world (cell) units.
    float yaw_      = 0.6f;   ///< Azimuth around +Y, radians.
    float pitch_    = 0.3f;   ///< Elevation, radians, clamped off the poles.
    float distance_ = 500.0f; ///< Eye-to-target distance, cells.

    // -- goal state: input handlers write here, springs chase it --
    Vec3f targetGoal_{0, 0, 0};
    float yawGoal_   = 0.6f;
    float pitchGoal_ = 0.3f;
    float distGoal_  = 500.0f;

    // -- spring velocities (one per smoothed degree of freedom) --
    Vec3f targetVel_{0, 0, 0};
    float yawVel_   = 0.0f;
    float pitchVel_ = 0.0f;
    float distVel_  = 0.0f;

    float sceneRadius_ = 500.0f; ///< Set by frameDomain; drives clip planes
                                 ///< and zoom clamping.
};

} // namespace foilcfd
