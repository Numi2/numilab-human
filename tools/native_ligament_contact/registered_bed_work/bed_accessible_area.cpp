#include "NumiHumanBedFootprint.hpp"
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

extern "C" double accessible_area(const double* xyz, unsigned count,
                                  const unsigned* faces, unsigned indices,
                                  double limit) noexcept {
    try {
        std::vector<numiHumanBedFootprint::Point> points;
        points.reserve(count);
        for (unsigned i=0;i<count;++i) {
            points.push_back({xyz[3*i],xyz[3*i+1],xyz[3*i+2]});
        }
        const auto weights=numiHumanBedFootprint::areas(points,
            std::vector<unsigned>(faces,faces+indices),std::vector<double>(count,limit));
        return std::accumulate(weights.begin(),weights.end(),0.0);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}
