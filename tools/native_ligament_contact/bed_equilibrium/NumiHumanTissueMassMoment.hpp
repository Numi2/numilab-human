#pragma once
#include <array>
#include <cmath>
#include <stdexcept>

// Accumulate the reference or accepted FEM snapshot in its world frame.
// Material reference coordinates are retained as a high/low displacement pair.
struct NumiHumanTissueMassMoment {
    double massKg = 0.0;
    std::array<double, 3> firstKgM{};
    template <class Node>
    void add(const Node& n) {
        const auto& p=n.positionAndMass;
        const auto& u=n.referenceDisplacementAndMode;
        const auto& l=n.referenceDisplacementLow;
        const auto& r=n.restAndFixed;
        const std::array<double,3> x = u.w==1.0f
            ? std::array<double,3>{double(r.x)+u.x+l.x,double(r.y)+u.y+l.y,double(r.z)+u.z+l.z}
            : std::array<double,3>{p.x,p.y,p.z};
        if (!(p.w>=0.0f && std::isfinite(p.w) && std::isfinite(x[0]) &&
              std::isfinite(x[1]) && std::isfinite(x[2])))
            throw std::runtime_error("nonfinite FEM mass or world coordinate");
        massKg+=p.w;
        for (unsigned k=0;k<3;++k) firstKgM[k]+=p.w*x[k];
    }
};
