#include <format>
#include <iostream>

#include "pcmath/pcmath.hpp"

std::string MatToString(const pcm::Mat2 &m) {
    return std::format("[[{}, {}],\n[{}, {}]]", m[0][0], m[1][0], m[0][1], m[1][1]);
}

std::string MatToString(const pcm::Mat3 &m) {
    return std::format("[[{}, {}, {}],\n[{}, {}, {}],\n[{}, {}, {}]]",
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2]
    );
}

std::string MatToString(const pcm::Mat4 &m) {
    return std::format("[[{}, {}, {}, {}],\n[{}, {}, {}, {}],\n[{}, {}, {}, {}],\n[{}, {}, {}, {}]]",
        m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
        m[0][3], m[1][3], m[2][3], m[3][3]
    );
}

void TestVec() {
    pcm::Vec2 v0(0.2f, 0.4f);
    pcm::Vec3 v1(v0, 3.0f);
    pcm::Vec4 v2(v1, 2.1f);

    std::cout << std::format("v0 = ({}, {})", v0.X(), v0.Y()) << std::endl;
    std::cout << std::format("v1 = ({}, {}, {})", v1.X(), v1.Y(), v1.Z()) << std::endl;
    std::cout << std::format("v2 = ({}, {}, {}, {})", v2.X(), v2.Y(), v2.Z(), v2.W()) << std::endl;

    pcm::Vec3 v3(0.3f, 0.5f, 1.0f);
    pcm::Vec3 v4(0.6f, 0.9f, 2.5f);
    pcm::Vec3 v5 = (v3 - v1).Cross(v4 - v1);
    std::cout << std::format("v5 = ({}, {}, {})", v5.X(), v5.Y(), v5.Z()) << std::endl;

    float len = (2.0f * (v4 - v3) + 1.3f * (v5 - v4)).Length();
    std::cout << std::format("len = {}", len) << std::endl;

    v1 = v5 + v3;
    std::cout << std::format("v1 (2) = ({}, {}, {})", v1.X(), v1.Y(), v1.Z()) << std::endl;

    pcm::Vec3 v6 = pcm::Vec3::UnitZ();
    std::cout << std::format("v6 = ({}, {}, {})", v6.X(), v6.Y(), v6.Z()) << std::endl;
}

void TestMat() {
    pcm::Mat2 m0 = pcm::Mat2::Identity();
    pcm::Mat3 m1 = pcm::Mat3::Identity() / 2.0f;
    pcm::Mat4 m2 = 2.2f * pcm::Mat4::Identity();
    std::cout << std::format("m0 = {}", MatToString(m0)) << std::endl;
    std::cout << std::format("m1 = {}", MatToString(m1)) << std::endl;
    std::cout << std::format("m2 = {}", MatToString(m2)) << std::endl;

    pcm::Mat3 m3 = pcm::Mat3(pcm::Vec3(10.0f, 1.0f, 1.0f), pcm::Vec3(2.0f, 10.0f, 2.0f), pcm::Vec3(3.0f, 3.0f, 10.0f));
    std::cout << std::format("m3 = {}", MatToString(m3)) << std::endl;

    pcm::Mat3 m4 = m3.Transpose();
    std::cout << std::format("m4 = {}", MatToString(m4)) << std::endl;

    pcm::Mat3 m5 = m3.Inverse();
    pcm::Mat3 m6 = m3 * m5;
    std::cout << std::format("m5 = {}", MatToString(m5)) << std::endl;
    std::cout << std::format("m6 = {}", MatToString(m6)) << std::endl;
}

void TestUtils() {
    pcm::Mat4 m0 = pcm::Perspective(pcm::Radians(45.0f), 1.3f, 0.001f, 100.0f);
    std::cout << std::format("m0 = {}", MatToString(m0)) << std::endl;
}

int main() {
    TestVec();

    TestMat();

    TestUtils();

    return 0;
}