// Orbit camera implementation: spherical orbit math, input mapping, the
// critically damped spring smoothing that makes orbit/pan/zoom glide, and the
// domain-framing heuristic that picks an initial distance from grid size.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include "camera.h"

#include <algorithm>
#include <cmath>

namespace foilcfd {

namespace {

constexpr float kPi = 3.14159265358979f;
// Pitch limit just short of the poles so lookAt's up vector never collapses.
constexpr float kPitchLimit = 0.49f * kPi;
// Input scaling: radians per pixel of drag, fraction per scroll notch.
constexpr float kRotatePerPixel = 0.005f;
constexpr float kPanPerPixel    = 0.0015f;
constexpr float kZoomPerNotch   = 0.10f;

// Spring stiffness as the natural frequency omega [rad/s]. Critical damping
// fixes the damping ratio at 1, so omega alone sets the feel: ~14 rad/s gives
// a ~150 ms settle — snappy but visibly smooth at 60 fps.
constexpr float kSpringOmega = 14.0f;

// Integrator safety: a debugger pause or load hitch can hand us a huge dt;
// clamping keeps the semi-implicit step well inside its stability region
// (stable for dt < 2/omega ~= 0.14 s).
constexpr float kMaxDt = 0.05f;

/// @brief One semi-implicit Euler step of a critically damped spring:
///   acc = omega^2 * (goal - x) - 2 * omega * v
/// Updating velocity before position keeps the scheme stable at game-loop dt.
/// @param x    Smoothed value (advanced in place).
/// @param v    Spring velocity (advanced in place).
/// @param goal Value the spring chases.
/// @param dt   Time step in seconds (pre-clamped by the caller).
void springStep(float& x, float& v, float goal, float dt) {
    const float acc = kSpringOmega * kSpringOmega * (goal - x)
                    - 2.0f * kSpringOmega * v;
    v += acc * dt;
    x += v * dt;
}

} // namespace

void OrbitCamera::frameDomain(int nx, int ny, int nz) {
    // Target the geometric center of the lattice box.
    targetGoal_ = Vec3f(static_cast<float>(nx) * 0.5f,
                        static_cast<float>(ny) * 0.5f,
                        static_cast<float>(nz) * 0.5f);
    // Scene radius = half the box diagonal; distance backs off enough that a
    // ~45 degree FOV sees the whole domain with margin.
    const Vec3f half(static_cast<float>(nx) * 0.5f,
                     static_cast<float>(ny) * 0.5f,
                     static_cast<float>(nz) * 0.5f);
    sceneRadius_ = length(half);
    distGoal_    = sceneRadius_ * 2.2f;
    // Grid rebuilds replace the whole world — animating across them would
    // sweep through garbage, so this entry point snaps.
    snapToGoals();
}

void OrbitCamera::frameRegion(const Vec3f& center, float radius) {
    targetGoal_ = center;
    // Distance for a 45 degree vertical FOV to contain a sphere of `radius`:
    // d = r / sin(fov/2), padded ~15% so the region doesn't kiss the edges.
    const float fitDist = radius / std::sin(0.5f * 45.0f * kPi / 180.0f);
    distGoal_ = std::clamp(fitDist * 1.15f,
                           0.05f * sceneRadius_, 20.0f * sceneRadius_);
    // Animated on purpose: "focus the foil" should glide, not teleport.
}

void OrbitCamera::rotate(float dxPixels, float dyPixels) {
    yawGoal_  += dxPixels * kRotatePerPixel;
    pitchGoal_ = std::clamp(pitchGoal_ + dyPixels * kRotatePerPixel,
                            -kPitchLimit, kPitchLimit);
}

void OrbitCamera::pan(float dxPixels, float dyPixels) {
    // Move the target in the camera's screen plane, distance-scaled so the
    // pan speed matches the apparent world speed at the target depth. The
    // basis comes from the GOAL state so consecutive drags compose exactly
    // even while the smoothed view is still catching up.
    const float cp = std::cos(pitchGoal_);
    const Vec3f fwd = Vec3f(-cp * std::cos(yawGoal_), -std::sin(pitchGoal_),
                            -cp * std::sin(yawGoal_)); // target - eye direction
    const Vec3f right = normalized(cross(fwd, Vec3f(0, 1, 0)));
    const Vec3f up    = cross(right, fwd);
    const float s = distGoal_ * kPanPerPixel;
    targetGoal_ = targetGoal_ + right * (-dxPixels * s) + up * (dyPixels * s);
}

void OrbitCamera::zoom(float scrollNotches) {
    // Exponential zoom feels uniform across scales.
    distGoal_ = std::clamp(
        distGoal_ * std::pow(1.0f - kZoomPerNotch, scrollNotches),
        0.05f * sceneRadius_, 20.0f * sceneRadius_);
}

void OrbitCamera::update(float dtSeconds) {
    const float dt = std::clamp(dtSeconds, 0.0f, kMaxDt);
    if (dt <= 0.0f) return;
    // One independent critically damped spring per degree of freedom. The
    // parameters are decoupled in spherical space, so per-axis springs stay
    // visually coherent (no curvature artifacts worth coupling them for).
    springStep(yaw_,      yawVel_,   yawGoal_,   dt);
    springStep(pitch_,    pitchVel_, pitchGoal_, dt);
    springStep(distance_, distVel_,  distGoal_,  dt);
    springStep(target_.x, targetVel_.x, targetGoal_.x, dt);
    springStep(target_.y, targetVel_.y, targetGoal_.y, dt);
    springStep(target_.z, targetVel_.z, targetGoal_.z, dt);
    // The smoothed pitch obeys the same pole clamp as the goal: the spring
    // can otherwise overshoot... it cannot (critically damped), but a snapped
    // goal right at the limit plus float error can graze it. Cheap insurance.
    pitch_    = std::clamp(pitch_, -kPitchLimit, kPitchLimit);
    distance_ = std::max(distance_, 1e-3f);
}

void OrbitCamera::snapToGoals() {
    target_   = targetGoal_;
    yaw_      = yawGoal_;
    pitch_    = pitchGoal_;
    distance_ = distGoal_;
    targetVel_ = Vec3f(0, 0, 0);
    yawVel_ = pitchVel_ = distVel_ = 0.0f;
}

Vec3f OrbitCamera::eye() const {
    // Spherical -> Cartesian around the target; yaw about +Y, pitch toward it.
    const float cp = std::cos(pitch_);
    return target_ + Vec3f(cp * std::cos(yaw_), std::sin(pitch_),
                           cp * std::sin(yaw_)) * distance_;
}

Mat4f OrbitCamera::viewMatrix() const {
    return Mat4f::lookAt(eye(), target_, Vec3f(0, 1, 0));
}

Mat4f OrbitCamera::projMatrix(float aspect) const {
    // Clip planes track the framed scene so depth precision follows zoom.
    const float nearZ = std::max(0.05f * sceneRadius_, 0.1f);
    const float farZ  = 10.0f * sceneRadius_ + distance_;
    return Mat4f::perspective(45.0f * kPi / 180.0f, aspect, nearZ, farZ);
}

void OrbitCamera::setYaw(float v) {
    yawGoal_ = v;
    yaw_     = v;
    yawVel_  = 0.0f;
}

void OrbitCamera::setPitch(float v) {
    pitchGoal_ = std::clamp(v, -kPitchLimit, kPitchLimit);
    pitch_     = pitchGoal_;
    pitchVel_  = 0.0f;
}

void OrbitCamera::setDistance(float v) {
    distGoal_ = std::clamp(v, 0.05f * sceneRadius_, 20.0f * sceneRadius_);
    distance_ = distGoal_;
    distVel_  = 0.0f;
}

void OrbitCamera::setTarget(const Vec3f& t) {
    targetGoal_ = t;
    target_     = t;
    targetVel_  = Vec3f(0, 0, 0);
}

} // namespace foilcfd
