#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace numiHumanCellShape {
using Vec = std::array<double, 3>;
inline double squaredDistance(const Vec& a, const Vec& b) {
    const double x=a[0]-b[0], y=a[1]-b[1], z=a[2]-b[2];
    return x*x+y*y+z*z;
}
inline double normalizedDeterminant(double det,const Vec& b,const Vec& c,const Vec& d) {
    const Vec origin{};
    const double longest2=std::max({squaredDistance(origin,b),squaredDistance(origin,c),
        squaredDistance(origin,d),squaredDistance(b,c),squaredDistance(b,d),squaredDistance(c,d)});
    return longest2>0.0 ? det/(longest2*std::sqrt(longest2)) : 0.0;
}
struct Summary {
    std::uint32_t cells=0, inverted=0, belowVolume=0, belowShape=0, nonfinite=0;
    std::uint32_t minimumShapeRow=0;
    double minimumShape=std::numeric_limits<double>::infinity();
    double minimumVolume=std::numeric_limits<double>::infinity();
    void observe(double det,double shape,std::uint32_t row) {
        ++cells;
        if(!std::isfinite(det)||!std::isfinite(shape)){++nonfinite;return;}
        const double volume=det/6.0;
        inverted+=volume<=0.0;
        belowVolume+=volume<=1.0e-18;
        belowShape+=shape<1.0e-5;
        minimumVolume=std::min(minimumVolume,volume);
        if(shape<minimumShape){minimumShape=shape;minimumShapeRow=row;}
    }
};
}
