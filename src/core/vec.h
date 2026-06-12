// Minimal host-side vector/matrix math: Vec2f, Vec3f, Mat4f and the handful of
// operations the geometry, camera, and renderer modules need. Deliberately tiny —
// the plan forbids a GLM dependency, and CUDA device code uses the builtin float3.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE
#pragma once

#include <cmath>

namespace foilcfd {

/// @brief 2D float vector for airfoil polygon / surface-frame math.
struct Vec2f {
    float x = 0.0f;
    float y = 0.0f;

    constexpr Vec2f() = default;
    constexpr Vec2f(float x_, float y_) : x(x_), y(y_) {}

    constexpr Vec2f operator+(const Vec2f& r) const { return {x + r.x, y + r.y}; }
    constexpr Vec2f operator-(const Vec2f& r) const { return {x - r.x, y - r.y}; }
    constexpr Vec2f operator*(float s) const { return {x * s, y * s}; }
    constexpr Vec2f operator/(float s) const { return {x / s, y / s}; }
};

/// @brief Dot product of two 2D vectors.
constexpr float dot(const Vec2f& a, const Vec2f& b) { return a.x * b.x + a.y * b.y; }

/// @brief Euclidean length of a 2D vector.
inline float length(const Vec2f& v) { return std::sqrt(dot(v, v)); }

/// @brief Unit-length copy of @p v (returns v unchanged if near-zero to avoid NaN).
inline Vec2f normalized(const Vec2f& v) {
    const float len = length(v);
    return (len > 1e-12f) ? v / len : v;
}

/// @brief 90-degree counter-clockwise rotation — turns a surface tangent into its
/// outward normal when the polygon winds clockwise around the foil interior.
constexpr Vec2f perpCCW(const Vec2f& v) { return {-v.y, v.x}; }

/// @brief Rotate @p v by @p radians about the origin (CCW positive).
inline Vec2f rotated(const Vec2f& v, float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return {v.x * c - v.y * s, v.x * s + v.y * c};
}

/// @brief 3D float vector for camera, mesh, and STL math (host code only).
struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    constexpr Vec3f() = default;
    constexpr Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    constexpr Vec3f operator+(const Vec3f& r) const { return {x + r.x, y + r.y, z + r.z}; }
    constexpr Vec3f operator-(const Vec3f& r) const { return {x - r.x, y - r.y, z - r.z}; }
    constexpr Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3f operator/(float s) const { return {x / s, y / s, z / s}; }
};

/// @brief Dot product of two 3D vectors.
constexpr float dot(const Vec3f& a, const Vec3f& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

/// @brief Cross product (right-handed).
constexpr Vec3f cross(const Vec3f& a, const Vec3f& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// @brief Euclidean length of a 3D vector.
inline float length(const Vec3f& v) { return std::sqrt(dot(v, v)); }

/// @brief Unit-length copy of @p v (returns v unchanged if near-zero to avoid NaN).
inline Vec3f normalized(const Vec3f& v) {
    const float len = length(v);
    return (len > 1e-12f) ? v / len : v;
}

/// @brief Column-major 4x4 matrix, matching OpenGL's glUniformMatrix4fv layout.
/// Element (row r, col c) lives at m[c * 4 + r].
struct Mat4f {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    /// @brief Identity matrix.
    static Mat4f identity() { return Mat4f{}; }

    /// @brief Matrix product `this * r` (column-major convention: applies r first).
    Mat4f operator*(const Mat4f& r) const {
        Mat4f out;
        // Standard column-major multiply: out[c][row] = sum_k this[k][row] * r[c][k].
        for (int c = 0; c < 4; ++c) {
            for (int row = 0; row < 4; ++row) {
                float acc = 0.0f;
                for (int k = 0; k < 4; ++k) acc += m[k * 4 + row] * r.m[c * 4 + k];
                out.m[c * 4 + row] = acc;
            }
        }
        return out;
    }

    /// @brief Right-handed look-at view matrix (OpenGL convention, -Z forward).
    /// @param eye    Camera position in world space.
    /// @param target Point the camera looks at.
    /// @param up     World-space up hint (normalized internally).
    static Mat4f lookAt(const Vec3f& eye, const Vec3f& target, const Vec3f& up) {
        // Build the orthonormal camera basis: f points from eye toward target.
        const Vec3f f = normalized(target - eye);
        const Vec3f s = normalized(cross(f, up));
        const Vec3f u = cross(s, f);
        Mat4f out;
        // Rotation rows are the basis vectors; translation is -basis . eye.
        out.m[0] = s.x;  out.m[4] = s.y;  out.m[8] = s.z;   out.m[12] = -dot(s, eye);
        out.m[1] = u.x;  out.m[5] = u.y;  out.m[9] = u.z;   out.m[13] = -dot(u, eye);
        out.m[2] = -f.x; out.m[6] = -f.y; out.m[10] = -f.z; out.m[14] = dot(f, eye);
        out.m[3] = 0;    out.m[7] = 0;    out.m[11] = 0;    out.m[15] = 1;
        return out;
    }

    /// @brief Right-handed perspective projection (OpenGL clip space, z in [-1,1]).
    /// @param fovyRadians Vertical field of view in radians.
    /// @param aspect      Viewport width / height.
    /// @param zNear       Near clip distance (> 0).
    /// @param zFar        Far clip distance (> zNear).
    static Mat4f perspective(float fovyRadians, float aspect, float zNear, float zFar) {
        const float t = std::tan(fovyRadians * 0.5f);
        Mat4f out;
        // Zero everything first — the default constructor builds identity.
        for (float& v : out.m) v = 0.0f;
        out.m[0]  = 1.0f / (aspect * t);
        out.m[5]  = 1.0f / t;
        out.m[10] = -(zFar + zNear) / (zFar - zNear);
        out.m[11] = -1.0f;
        out.m[14] = -(2.0f * zFar * zNear) / (zFar - zNear);
        return out;
    }
};

} // namespace foilcfd
