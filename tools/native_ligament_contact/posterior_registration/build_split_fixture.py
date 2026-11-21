import sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:])
dst.mkdir(exist_ok=True)
g=(src/'pelvic_end_geometry.metalinc').read_text()
split=g[g.index('struct PelvicSplitPoint'):g.index('inline uint4 pelvicJoinWitness')]
(dst/'split-under-test.metalinc').write_text(split)
(dst/'test-split.metal').write_text('''#include <metal_stdlib>
using namespace metal;
#include "pelvic_crossing_predicates.metalinc"
#include "split-under-test.metalinc"
kernel void testGeometrySigns(device const float4* data [[buffer(0)]],device int4* out [[buffer(1)]],uint i [[thread_position_in_grid]]){
    PelvicSplitPoint a[3],b[3],v[4];
    for(uint j=0;j<3;++j){
        uint k=12*i+2*j;
        a[j]={data[k].xyz,data[k+1].xyz};b[j]={data[k+6].xyz,data[k+7].xyz};v[j]=a[j];
    }
    v[3]=b[0];out[i]=int4(pelvicSplitOrient(v[0],v[1],v[2],v[3]),pelvicSplitExact(v),pelvicSplitCross(a,b),0);
}
''')
s=Path('verify_geometry_signs.mm').read_text().replace('%96','%192').replace('/96','/192')
(dst/'verify.mm').write_text(s)
