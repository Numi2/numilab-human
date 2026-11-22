#include "fem_reference_geometry.hpp"
#include <cmath>
using namespace numi::matter::detail;
extern "C" int checkEdge(float ah,float al,float bh,float bl,float* value){
    if (!femExactDisplacementEdge(ah,al,bh,bl,*value)) return 0;
    const auto shader=mrCompensatedAdd({ah,al},{-bh,-bl});
    return shader.high==*value && shader.low==0 ? 1 : -1;
}
extern "C" int checkCell(){
    NMFEMNodeStateGPU nodes[4]{};
    NMTetrahedronGPU t{};
    t.nodes={0,1,2,3};
    t.identity.w=NM_TETRAHEDRON_REFERENCE_DISPLACEMENT_GRADIENT;
    t.inverseRestRow0.x=1048576.0f;
    t.inverseRestRow1.y=134217728.0f;
    t.inverseRestRow2.z=134217728.0f;
    for(auto& n:nodes)n.referenceDisplacementAndMode={0,0,-0.001953125f,1};
    const nm_float4 validity={0.35f,2.5f,0,0};
    const auto rest=femReferenceDeterminantInterval(t,nodes);
    if(rest.nominal!=1.0f || !rest.strictlyAdmitted(validity)) return 1;
    nodes[3].referenceDisplacementAndMode.z -= 1.0f/134217728.0f;
    const auto collapsed=femReferenceDeterminantInterval(t,nodes);
    if(collapsed.strictlyAdmitted(validity) || collapsed.nominal!=0.0f) return 2;
    nodes[3].referenceDisplacementAndMode.z -= 1.0f/134217728.0f;
    const auto inverted=femReferenceDeterminantInterval(t,nodes);
    if(inverted.strictlyAdmitted(validity) || inverted.nominal!=-1.0f) return 3;
    return 0;
}
