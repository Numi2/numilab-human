"""Cross-check the emitted Metal fast path and parent/child gate against rational membership."""
from fractions import Fraction as F
from pathlib import Path
import json,struct,sys,subprocess
import numpy as np

def inside(point, bounds):
    q=list(map(lambda x:F(float(x)),point))
    rows=[[F(float(x)) for x in row] for row in bounds['tube']]
    a,b,lo,hi=rows;p=bounds['pivot'];t=(q[p]-a[p])/(b[p]-a[p])
    return 0<=t<=1 and all(lo[j]<=q[j]-a[j]-t*(b[j]-a[j])<=hi[j] for j in range(7))
def test(work,src,app,compiled):
    records=json.loads((work/'four-domains.json').read_text())['domains']
    proofs=json.loads((work/'refinement.json').read_text())['parents']
    # Reuse the existing Metal dispatch harness; it indexes one result per input.
    shader='''#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "cardiac_certified_tube.metalinc"
#include "cardiac_proved_table.metalinc"
#include "native_cardiac_subdomains.metalinc"
kernel void test_cardiac_tubes(device const MRHumanRestingCommonCoordinatesGPU* states [[buffer(0)]],
    constant CardiacCertifiedTube* unused [[buffer(1)]],
    constant uint& count [[buffer(2)]],device uint* result [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    const auto state=states[i];const float q[7]={state.first.x,state.first.y,state.first.z,state.first.w,
        state.second.x,state.second.y,state.second.z};
    uint id=0,code=MR_INVALID_INDEX;
    if(state.status.x==0u) code=0;
    else if(state.status.x==5u) {
        for(uint j=0;j<sizeof(cardiacCertifiedTubes)/sizeof(cardiacCertifiedTubes[0]);++j)
            if(cardiacInsideCertifiedTube(cardiacCertifiedTubes[j],q)){code=1u;break;}
        if(code==MR_INVALID_INDEX && cardiacNestedDomain(q,id)) code=2u+id;
    }
    result[i]=code;
}'''
    srcfile=compiled/'test-membership.metal';srcfile.write_text(shader)
    air=compiled/'test-membership.air';lib=compiled/'test-membership.metallib'
    subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-fno-fast-math','-I',str(app/'include'),'-c',srcfile,'-o',air],check=True)
    subprocess.run(['xcrun','-sdk','macosx','metallib',air,'-o',lib],check=True)
    rows=np.fromfile(work/'coordinates.bin','<f4').reshape(-1,16);assert len(rows)==30000
    rng=np.random.default_rng(8200)
    trials=[(r,f) for r in records for f in [-.05,0,.25,.5,.75,1,1.05]]
    rnd=np.zeros((len(trials)*12+5,16),dtype='<f4');rnd.view('<u4')[:,8]=5
    for i,(domain,f) in enumerate(trials):
        a,b,lo,hi=np.array(domain['tube']);g=rng.uniform(-.3,1.3,(12,7))
        rnd[12*i:12*(i+1),:7]=a+f*(b-a)+lo+g*(hi-lo)
    rnd[-5,:7]=np.nan;rnd[-4,:7]=np.inf;rnd[-3,:7]=-np.inf;rnd[-2,:7]=1.;rnd[-1]=rows[8200];rnd.view('<u4')[-1,8]=4
    stacked=np.vstack([rows,rnd]);target=work/'gate-inputs.bin';stacked.tofile(target)
    dummy=work/'unused-tube.bin';dummy.write_bytes(bytes(128));dest=work/'gate-results.bin'
    subprocess.run([src/'cardiac-tube-membership-probe',lib,target,dummy,dest],check=True,stdout=subprocess.DEVNULL)
    enc=np.fromfile(dest,'<u4');assert len(enc)==len(stacked)
    reject=0xffffffff
    assert (enc[:len(rows)]!=reject).all() and (enc[-5:]==reject).all()
    locations=[(p,None) for p in range(len(proofs))]
    locations.extend((p,c) for p,r in enumerate(proofs) for c in r['children'])
    for point,id in zip(stacked,enc,strict=True):
        code=int(id)
        if code in (0,1,reject):continue
        p,c=locations[code-2]
        assert inside(point[:7],records[p])
        assert proofs[p]['parent_accepted_without_children'] if c is None else inside(point[:7],c)
    summary=dict(forecast_states=len(rows),old_certified_exterior=int((enc[:len(rows)]==1).sum()),new_nested_admissions=int(((enc[:len(rows)]>=2)&(enc[:len(rows)]!=reject)).sum()),uncovered=0,
                 adversarial_points=len(rnd),adversarial_rejections=int((enc[-len(rnd):]==reject).sum()),every_new_admission_exact_parent_and_leaf=True)
    (work/'gpu-membership.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)
if __name__=='__main__':test(*map(Path,sys.argv[1:]))
