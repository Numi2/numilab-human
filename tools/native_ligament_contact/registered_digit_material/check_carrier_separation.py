"""Compare native tetrahedron separation against unbounded integer arithmetic."""

import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CPP = r"""
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include "NumiHumanRegisteredSurfaceIntersection.hpp"
using V3=std::array<double,3>;
using Tet=std::array<std::uint32_t,4>;
constexpr unsigned faces[][3]={{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
constexpr unsigned edges[][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
#include "NumiHumanCarrierSeparation.hpp"
int main() {
    std::array<std::array<float,3>,8> raw{};
    while(std::cin.read(reinterpret_cast<char*>(raw.data()),sizeof(raw))) {
        std::array<V3,8> p{};
        for(unsigned i=0;i<8;++i)for(unsigned a=0;a<3;++a)p[i][a]=raw[i][a];
        std::cout.put(disjointCarrierInteriors(p,{0,1,2,3},{4,5,6,7})?1:0);
    }
}
"""


def oracle(points):
    ratios = [[float(v).as_integer_ratio() for v in row] for row in points]
    scale = max(s for row in ratios for _, s in row)
    p = [tuple(n * (scale // s) for n, s in row) for row in ratios]

    def sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    def cross(u, v):
        return tuple(
            u[(i + 1) % 3] * v[(i + 2) % 3] - u[(i + 2) % 3] * v[(i + 1) % 3]
            for i in range(3)
        )

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v))

    left, right = p[:4], p[4:]
    pairs = lambda t: [sub(a, b) for a, b in itertools.combinations(t, 2)]
    axes = [
        cross(sub(b, a), sub(c, a))
        for tet in (left, right)
        for a, b, c in itertools.combinations(tet, 3)
    ]
    axes.extend(cross(a, b) for a in pairs(left) for b in pairs(right))
    for axis in axes:
        if any(axis):
            a = [dot(axis, v) for v in left]
            b = [dot(axis, v) for v in right]
            if max(a) <= min(b) or max(b) <= min(a):
                return 1
    return 0


def check(owner):
    rng = np.random.default_rng(82345)
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="f4")
    b = a.copy()
    b[3, 2] = -1
    cases = [
        np.vstack((a, b)),
        np.vstack((a, a)),
        np.vstack((a, a + 2)),
        np.vstack((a, a + [0.25, 0, 0])),
    ]
    for _ in range(300):
        p = rng.normal(size=(8, 3)).astype("f4")
        p[rng.integers(8), rng.integers(3)] = np.ldexp(1.0, -120)
        scale = np.exp2(rng.integers(-25, 26, size=3))
        cases.append((p * scale).astype("f4"))
    cases = np.stack(cases).astype("f4")
    expected = bytes(map(oracle, cases))
    assert list(expected[:4]) == [1, 0, 1, 0]
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "check.cpp"
        binary = Path(work) / "check"
        source.write_text(CPP)
        subprocess.run(
            [
                "xcrun",
                "clang++",
                "-std=c++20",
                "-O3",
                "-I",
                str(owner),
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
        )
        actual = subprocess.run(
            [str(binary)], input=cases.tobytes(), capture_output=True, check=True
        ).stdout
    if actual != expected:
        raise ValueError(
            "Native separating planes disagree with exact integer arithmetic"
        )
    print(
        f"{len(cases)} exact comparisons passed, {sum(expected)} disjoint and {len(expected) - sum(expected)} overlapping"
    )


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
