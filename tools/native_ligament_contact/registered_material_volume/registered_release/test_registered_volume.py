"""Compare the production GPU predicates against exact rational arithmetic at a strict volume floor."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,subprocess,sys
import numpy as np

def test(shader,volume,fit,runner,out):
    out.mkdir()
    h=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    meta=json.loads((shader/'build.json').read_bytes())
    source=shader/'src/metal/fem.metalinc'
    if h(source)!=meta['fem']: raise ValueError('Unmatched shader source')
    s=source.read_text()
    body=s[s.index('inline MRCompensatedScalar registeredNeg('):s.index('kernel void nm_fem_parallel_backtracking_tiles(')]
    code='#include <metal_stdlib>\nusing namespace metal;\n#include "numi/matter/shared.h"\n#include "metalrobo/compensated_translation_gpu.h"\n'+body+'''
kernel void registered_volume_test(device const NMFEMNodeStateGPU* v [[buffer(0)]],device const float4* b [[buffer(1)]],device float4* out [[buffer(2)]],uint id [[thread_position_in_grid]]){
   uint4 ix=4u*id+uint4(0,1,2,3);
   NMFEMNodeStateGPU n[4];for(uint i=0;i<4;++i)n[i]=v[ix[i]];
   auto volume=registeredVolumeWord(v,ix);
   auto trial=registeredReleaseMargin(registeredStateVolumeWord(n),b[id]);
   auto terminal=registeredReleaseMargin(volume,b[id]);
   out[id]=float4(float(registeredReleasePass(trial,b[id])),float(registeredReleasePass(terminal,b[id])),volume.high,volume.low);
}
'''
    metal=out/'test.metal';metal.write_text(code); cmd=meta['commands'][0]
    args=cmd[:cmd.index('-c')]+['-c',str(metal),'-o',str(out/'test.air')]
    comp=meta['commands'][1][:4]+[str(out/'test.air'),'-o',str(out/'test.metallib')]
    def run(cmd):subprocess.run(cmd,check=True)
    run(args);run(comp);run(['xcrun','clang++','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(runner),'-o',str(out/'test')])
    nodes=[];floors=[];names=[];expected=[]
    F=lambda x:Fraction(float(x))
    def add(name,ref,disp,low,flag,threshold=None,delta=0):
        n=np.zeros((4,6,4),'f4');n[:,2,:3]=ref;n[:,4,:3]=disp;n[:,5,:3]=low;n[:,4,3]=flag
        n[:,0,:3]=ref.astype('f8')+disp.astype('f8')+low.astype('f8')
        def value(i,k):
            return sum(F(n[i,j,k]) for j in (2,4,5)) if all(n[:,4,3]==1) else F(n[i,0,k])
        a,b,c=[[value(i,k)-value(0,k) for k in range(3)] for i in (1,2,3)]
        det=sum(a[k]*(b[(k+1)%3]*c[(k+2)%3]-b[(k+2)%3]*c[(k+1)%3])for k in range(3))
        center=det if threshold is None else F(threshold)
        high=np.float32(float(center));lo=np.float32(float(center-F(high)))
        # For actual non-dyadic determinants, the two-limb floor is rounded.
        # One relative FP32 ulp gives a resolvable offset in either direction.
        if delta:lo=np.float32(lo+delta*abs(np.spacing(high)))
        floor=F(high)+F(lo);orientation=1 if det>0 else -1
        nodes.append(n);floors.append([high,lo,.35,orientation]);names.append(name)
        expected.append((det-floor)*orientation>0)
    zeros=np.zeros((4,3),'f4');std=np.concatenate((np.zeros((1,3)),np.eye(3))).astype('f4')
    for o in (1,-1):
        v=std.copy();v[3,2]*=o
        for delta in (-1,0,1):
            for flag in ([0]*4,[1]*4,[1,1,0,1]):
                add(f'unit_{o}_{delta}_{flag}',v+16,zeros,zeros,flag,delta=delta)
        u=zeros.copy();u[3,2]=o*2.**-26
        for sign in (-1,1):
            add(f'low_displacement_{o}_{sign}',v+16,zeros,u*sign,[1]*4,threshold=float(o))
    ids={1311504,1293813}
    cells=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')[sorted(ids)]-1
    x=np.loadtxt(volume/'mesh.node',skiprows=1,usecols=(1,2,3));y=np.loadtxt(fit/'current.node',skiprows=1,usecols=(1,2,3))
    for i,c in zip(sorted(ids),cells):
        v=x[c].astype('f4');d=(y[c]-v).astype('f4');lo=(y[c]-v-d).astype('f4')
        for delta in (-4,4):add(f'fat_{i}_{delta}',v,d,lo,[1]*4,delta=delta)
    np.asarray(nodes,'<f4').tofile(out/'nodes.f32');np.asarray(floors,'<f4').tofile(out/'floor.f32')
    run([str(out/'test'),*[str(out/name) for name in ('test.metallib','nodes.f32','floor.f32','observed.f32')]])
    results=np.fromfile(out/'observed.f32','<f4').reshape(-1,4)
    wrong=[name for name,row,p in zip(names,results,expected) if row[0]!=float(p) or row[1]!=float(p)]
    meta=dict(tests=len(names),rejected=sum(not p for p in expected),accepted=sum(expected),failures=wrong,source_sha256=h(source),kernel_sha256=h(metal),device=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),cases=names)
    (out/'test.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(meta['tests'],len(wrong))
    if wrong:raise ValueError(wrong)
if __name__=='__main__':test(*(Path(a).resolve() for a in sys.argv[1:]))
