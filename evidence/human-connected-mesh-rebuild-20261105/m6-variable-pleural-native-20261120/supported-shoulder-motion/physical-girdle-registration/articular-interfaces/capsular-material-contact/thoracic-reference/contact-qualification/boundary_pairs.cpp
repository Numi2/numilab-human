#include "NumiHumanRegisteredSurfaceIntersection.hpp"
#include <algorithm>
#include <limits>

namespace surface = metalrobo::human::coarse_tissue::registered_surface_intersection_detail;

// Only reject if every nonshared point has a certified strict sign. Shared
// material corners are exactly on the plane by construction. Ambiguous signs
// and coplanar configurations always use the existing expansion predicates.
static bool planeSeparates(const surface::Triangle& a, const surface::Indices& ai,
                           const surface::Triangle& b, const surface::Indices& bi) {
    int side = 0;
    for (unsigned corner=0; corner<3; ++corner) {
        if (std::find(ai.begin(), ai.end(), bi[corner]) != ai.end()) continue;
        surface::Point u,v,w;
        for (unsigned k=0; k<3; ++k) {
            u[k]=a[1][k]-a[0][k]; v[k]=a[2][k]-a[0][k]; w[k]=b[corner][k]-a[0][k];
        }
        double det=0., bound=0.;
        for(unsigned i=0; i<3; ++i) {
            const auto j=(i+1)%3, k=(i+2)%3;
            const double x=v[j]*w[k], y=v[k]*w[j];
            det+=u[i]*(x-y);
            bound+=std::abs(u[i])*(std::abs(x)+std::abs(y));
        }
        if (std::abs(det)<=32*std::numeric_limits<double>::epsilon()*bound) return false;
        const int s=det>0 ? 1 : -1;
        if (side && side!=s) return false;
        side=s;
    }
    return side!=0;
}

extern "C" void batch(const double* xyz, const unsigned* faces,
                      const unsigned* pairs, unsigned count, unsigned char* hit) {
    for(unsigned p=0;p<count;++p) {
        surface::Triangle a,b;
        surface::Indices ai,bi;
        for(unsigned k=0;k<3;++k) {
            ai[k]=faces[3*pairs[2*p]+k]; bi[k]=faces[3*pairs[2*p+1]+k];
            for(unsigned j=0;j<3;++j) {
                a[k][j]=xyz[3*ai[k]+j]; b[k][j]=xyz[3*bi[k]+j];
            }
        }
        hit[p]=!planeSeparates(a,ai,b,bi) && !planeSeparates(b,bi,a,ai)
            && surface::intersectsBeyondSharedFeatures(a,ai,b,bi);
    }
}
