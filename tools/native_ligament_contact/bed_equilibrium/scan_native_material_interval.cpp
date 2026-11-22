#include "fem_reference_geometry.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
using namespace numi::matter::detail;
extern "C" int scanCellIntervals(const double* rest, const double* current, size_t count,
                                  const int* cells, size_t numCells, double* out) {
    std::vector<NMFEMNodeStateGPU> nodes(count);
    for(size_t i=0;i<count;++i){
        auto& n=nodes[i];
        float high[3],low[3];
        for(unsigned j=0;j<3;++j){
            const double delta=current[3*i+j]-rest[3*i+j];
            high[j]=float(delta);low[j]=float(delta-double(high[j]));
        }
        n.referenceDisplacementAndMode={high[0],high[1],high[2],1};n.referenceDisplacementLow={low[0],low[1],low[2],0};
    }
    for(size_t i=0;i<numCells;++i){
        const int* s=cells+4*i;
        for(unsigned k=0;k<4;++k)if(s[k]<0 || size_t(s[k])>=count)return 1;
        double m[9];
        for(unsigned j=0;j<3;++j)for(unsigned k=0;k<3;++k)m[3*j+k]=double(float(rest[3*s[k+1]+j]))-double(float(rest[3*s[0]+j]));
        const double a=m[0],b=m[1],c=m[2],d=m[3],e=m[4],f=m[5],g=m[6],h=m[7],l=m[8];
        const double det=a*(e*l-f*h)-b*(d*l-f*g)+c*(d*h-e*g);
        if(det<=6.e-18)return 2;
        const double v=1.0/det;
        NMTetrahedronGPU t{};
        t.identity.w=NM_TETRAHEDRON_REFERENCE_DISPLACEMENT_GRADIENT;
        t.nodes={unsigned(s[0]),unsigned(s[1]),unsigned(s[2]),unsigned(s[3])};
        t.inverseRestRow0={float((e*l-f*h)*v),float((c*h-b*l)*v),float((b*f-c*e)*v),0};
        t.inverseRestRow1={float((f*g-d*l)*v),float((a*l-c*g)*v),float((c*d-a*f)*v),0};
        t.inverseRestRow2={float((d*h-e*g)*v),float((b*g-a*h)*v),float((a*e-b*d)*v),0};
        auto interval=femReferenceDeterminantInterval(t,nodes.data());
        out[3*i]=interval.center;out[3*i+1]=interval.radius;
        out[3*i+2]=interval.finite&&interval.noIntermediateOverflow ? interval.nominal : NAN;
    }
    return 0;
}
