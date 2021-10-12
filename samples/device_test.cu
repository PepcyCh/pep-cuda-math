#include <cstdio>
#include <vector>

#include "pcmath/pcmath.hpp"

__device__ void PrintMat2(const pcm::Mat2 &m, const char *name) {
    printf("%s = [[%f, %f],\n[%f, %f]]\n", name, m[0][0], m[1][0], m[0][1], m[1][1]);
}

__device__ void PrintMat3(const pcm::Mat3 &m, const char *name) {
    printf("%s = [[%f, %f, %f],\n[%f, %f, %f],\n[%f, %f, %f]]\n", name,
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2]
    );
}

__device__ void PrintMat4(const pcm::Mat4 &m, const char *name) {
    printf("%s = [[%f, %f, %f, %f],\n[%f, %f, %f, %f],\n[%f, %f, %f, %f],\n[%f, %f, %f, %f]]\n",
        name,
        m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
        m[0][3], m[1][3], m[2][3], m[3][3]
    );
}

__global__ void TestVec() {
    pcm::Vec2 v0(0.2f, 0.4f);
    pcm::Vec3 v1(v0, 3.0f);
    pcm::Vec4 v2(v1, 2.1f);

    printf("v0 = (%f, %f)\n", v0.X(), v0.Y());
    printf("v1 = (%f, %f, %f)\n", v1.X(), v1.Y(), v1.Z());
    printf("v2 = (%f, %f, %f, %f)\n", v2.X(), v2.Y(), v2.Z(), v2.W());

    pcm::Vec3 v3(0.3f, 0.5f, 1.0f);
    pcm::Vec3 v4(0.6f, 0.9f, 2.5f);
    pcm::Vec3 v5 = (v3 - v1).Cross(v4 - v1);
    printf("v5 = (%f, %f, %f)\n", v5.X(), v5.Y(), v5.Z());

    float len = (2.0f * (v4 - v3) + 1.3f * (v5 - v4)).Length();
    printf("len = %f\n", len);

    v1 = v5 + v3;
    printf("v1 (2) = (%f, %f, %f)\n", v1.X(), v1.Y(), v1.Z());

    pcm::Vec3 v6 = pcm::Vec3::UnitZ();
    printf("v6 = (%f, %f, %f)\n", v6.X(), v6.Y(), v6.Z());
}

__global__ void TestMat() {
    pcm::Mat2 m0 = pcm::Mat2::Identity();
    pcm::Mat3 m1 = pcm::Mat3::Identity() / 2.0f;
    pcm::Mat4 m2 = 2.2f * pcm::Mat4::Identity();
    PrintMat2(m0, "m0");
    PrintMat3(m1, "m1");
    PrintMat4(m2, "m2");

    pcm::Mat3 m3 = pcm::Mat3(pcm::Vec3(10.0f, 1.0f, 1.0f), pcm::Vec3(2.0f, 10.0f, 2.0f), pcm::Vec3(3.0f, 3.0f, 10.0f));
    PrintMat3(m3, "m3");

    pcm::Mat3 m4 = m3.Transpose();
    PrintMat3(m4, "m4");

    pcm::Mat3 m5 = m3.Inverse();
    pcm::Mat3 m6 = m3 * m5;
    PrintMat3(m5, "m5");
    PrintMat3(m6, "m6");
}

__global__ void TestUtils() {
    pcm::Mat4 m0 = pcm::Perspective(pcm::Radians(45.0f), 1.3f, 0.001f, 100.0f);
    PrintMat4(m0, "m0");
}

int main() {
    TestVec<<<1, 1>>>();

    TestMat<<<1, 1>>>();

    TestUtils<<<1, 1>>>();

    return 0;
}