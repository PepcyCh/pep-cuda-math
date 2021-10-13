#pragma once

#include "mat.hpp"

namespace pep::cmath {

CUDA_HOST_DEVICE inline float Radians(float degree) {
    return degree / 180.0f * 3.141592653589793238463f;
}
CUDA_HOST_DEVICE inline double Radians(double degree) {
    return degree / 180 * 3.141592653589793238463;
}
CUDA_HOST_DEVICE inline float Degree(float radians) {
    return radians * 0.3183098861837907f * 180;
}
CUDA_HOST_DEVICE inline double Degree(double radians) {
    return radians * 0.3183098861837907 * 180;
}

CUDA_HOST_DEVICE inline float Lerp(float a, float b, float t) {
    return a + (b - a) * t;
}
CUDA_HOST_DEVICE inline double Lerp(double a, double b, double t) {
    return a + (b - a) * t;
}

CUDA_HOST_DEVICE inline Mat4 Translate(const Vec3 &v) {
    return Mat4(
        Vec4(1.0f, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, 1.0f, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, 1.0f, 0.0f),
        Vec4(v, 1.0f)
    );
}
CUDA_HOST_DEVICE inline Mat4 Translate(float x, float y, float z) {
    return Mat4(
        Vec4(1.0f, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, 1.0f, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, 1.0f, 0.0f),
        Vec4(x, y, z, 1.0f)
    );
}

CUDA_HOST_DEVICE inline Mat4 Scale(const Vec3 &v) {
    return Mat4(
        Vec4(v.X(), 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, v.Y(), 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, v.Z(), 0.0f),
        Vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}
CUDA_HOST_DEVICE inline Mat4 Scale(float x, float y, float z) {
    return Mat4(
        Vec4(x, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, y, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, z, 0.0f),
        Vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

CUDA_HOST_DEVICE inline Mat4 RotateX(float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Mat4(
        Vec4(1.0f, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, c, s, 0.0f),
        Vec4(0.0f, -s, c, 0.0f),
        Vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}
CUDA_HOST_DEVICE inline Mat4 RotateY(float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Mat4(
        Vec4(c, 0.0f, -s, 0.0f),
        Vec4(0.0f, 1.0f, 0.0f, 0.0f),
        Vec4(s, 0.0f, c, 0.0f),
        Vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}
CUDA_HOST_DEVICE inline Mat4 RotateZ(float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Mat4(
        Vec4(c, s, 0.0f, 0.0f),
        Vec4(-s, c, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, 1.0f, 0.0f),
        Vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}
CUDA_HOST_DEVICE inline Mat4 Rotate(const Vec3 &axis, float angle) {
    const Vec3 a = axis.Normalize();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const float mc = 1.0f - c;
    return Mat4(
        Vec4(c + a[0] * a[0] * mc, a[0] * a[1] * mc + a[2] * s, a[2] * a[0] * mc - a[1] * s, 0.0f),
        Vec4(a[0] * a[1] * mc - a[2] * s, c + a[1] * a[1] * mc, a[1] * a[2] * mc + a[0] * s, 0.0f),
        Vec4(a[2] * a[0] * mc + a[1] * s, a[1] * a[2] * mc - a[0] * s, c + a[2] * a[2] * mc, 0.0f),
        Vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

CUDA_HOST_DEVICE inline Mat4 Rotate(const Vec4 &quaternion) {
    const float x2 = 2.0f * quaternion.X();
    const float y2 = 2.0f * quaternion.Y();
    const float z2 = 2.0f * quaternion.Z();
    const float xx = x2 * quaternion.X();
    const float yy = y2 * quaternion.Y();
    const float zz = z2 * quaternion.Z();
    const float xy = x2 * quaternion.Y();
    const float yz = y2 * quaternion.Z();
    const float zx = z2 * quaternion.X();
    const float xw = x2 * quaternion.W();
    const float yw = y2 * quaternion.W();
    const float zw = z2 * quaternion.W();
    return Mat4(
        Vec4(1.0f - yy - zz, xy + zw, zx - yw, 0.0f),
        Vec4(xy - zw, 1.0f - zz - xx, yz + xw, 0.0f),
        Vec4(zx + yw, yz - xw, 1.0f - xx - yy, 0.0f),
        Vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

CUDA_HOST_DEVICE inline Mat4 LookAt(const Vec3 &pos, const Vec3 &look_at, const Vec3 &up) {
    const Vec3 w = (pos - look_at).Normalize();
    const Vec3 u = up.Cross(w).Normalize();
    const Vec3 v = w.Cross(u);
    return Mat4(
        Vec4(u.X(), v.X(), w.X(), 0.0f),
        Vec4(u.Y(), v.Y(), w.Y(), 0.0f),
        Vec4(u.Z(), v.Z(), w.Z(), 0.0f),
        Vec4(-u.Dot(pos), -v.Dot(pos), -w.Dot(pos), 1.0f)
    );
}

CUDA_HOST_DEVICE inline Mat4 Perspective(float fov, float aspect, float n, float f, bool flip_y = false) {
    const float t = 1.0f/ std::tan(fov * 0.5f);
    const float invz = 1.0f / (f - n);
#ifdef PCMATH_GL_MATRIX
    Mat4 res(
        Vec4(t / aspect, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, t, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, -(f + n) * invz, -1.0f),
        Vec4(0.0f, 0.0f, -2.0f * f * n * invz, 0.0f)
    );
#else
    Mat4 res(
        Vec4(t / aspect, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, t, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, -f * invz, -1.0f),
        Vec4(0.0f, 0.0f, -f * n * invz, 0.0f)
    );
#endif
    if (flip_y) {
        res[1][1] = -res[1][1];
    }
    return res;
}

CUDA_HOST_DEVICE inline Mat4 PerspectiveInf(float fov, float aspect, float n, bool flip_y = false) {
    const float t = 1.0f/ std::tan(fov * 0.5f);
#ifdef PCMATH_GL_MATRIX
    Mat4 res(
        Vec4(t / aspect, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, t, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, -1.0f, -1.0f),
        Vec4(0.0f, 0.0f, -2.0f * n, 0.0f)
    );
#else
    Mat4 res(
        Vec4(t / aspect, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, t, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, -1.0f, -1.0f),
        Vec4(0.0f, 0.0f, -n, 0.0f)
    );
#endif
    if (flip_y) {
        res[1][1] = -res[1][1];
    }
    return res;
}

CUDA_HOST_DEVICE inline Mat4 PerspectiveReverseZ(float fov, float aspect, float n, float f, bool flip_y = false) {
    const float t = 1.0f / std::tan(fov * 0.5f);
    const float invz = 1.0f / (f - n);
    Mat4 res(
        Vec4(t / aspect, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, t, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, n * invz, -1.0f),
        Vec4(0.0f, 0.0f, f * n * invz, 0.0f)
    );
    if (flip_y) {
        res[1][1] = -res[1][1];
    }
    return res;
}

CUDA_HOST_DEVICE inline Mat4 PerspectiveInfReverseZ(float fov, float aspect, float n, bool flip_y = false) {
    const float t = 1.0f / std::tan(fov * 0.5f);
    Mat4 res(
        Vec4(t / aspect, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, t, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, 0.0f, -1.0f),
        Vec4(0.0f, 0.0f, n, 0.0f)
    );
    if (flip_y) {
        res[1][1] = -res[1][1];
    }
    return res;
}

CUDA_HOST_DEVICE inline Mat4 Orthographic(float l, float r, float b, float t, float n, float f, bool flip_y = false) {
    const float invw = 1.0f / (r - l);
    const float invh = 1.0f / (t - b);
    const float invd = 1.0f / (f - n);
#ifdef PCMATH_GL_MATRIX
    Mat4 res(
        Vec4(2.0f * invw, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, 2.0f * invh, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, -invd, 0.0f),
        Vec4(-(l + r) * invw, -(b + t) * invh, -n * invd, 1.0f)
    );
#else
    Mat4 res(
        Vec4(2.0f * invw, 0.0f, 0.0f, 0.0f),
        Vec4(0.0f, 2.0f * invh, 0.0f, 0.0f),
        Vec4(0.0f, 0.0f, -2.0f * invd, 0.0f),
        Vec4(-(l + r) * invw, -(b + t) * invh, -(n + f) * invd, 1.0f)
    );
#endif
    if (flip_y) {
        res[1][1] = -res[1][1];
    }
    return res;
}

}