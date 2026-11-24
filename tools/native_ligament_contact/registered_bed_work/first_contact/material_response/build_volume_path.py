"""Use the determinant path of flagged cells to condition a nonlinear search direction."""
from pathlib import Path
import json, os, shutil, subprocess, sys
from compile_connected_joint import digest

def run(base,out):
    src=base/'metal/src';dst=out/'metal/src';(dst/'metal').mkdir(parents=True)
    for f in src.iterdir():
        if f.name=='metal':continue
        if f.name=='matter_kernels.metal':shutil.copy2(f,dst/f.name)
        else:os.symlink(f.resolve(),dst/f.name)
    for f in (src/'metal').iterdir():
        if f.name!='fem.metalinc':os.symlink(f.resolve(),dst/'metal'/f.name)
    path=src/'metal/fem.metalinc';s=path.read_text();begin=s.index('    float3 gRef[4];',s.index('kernel void nm_fem_registered_volume_active'))
    end=s.index('\n}',begin)
    code=r'''    float3 gRef[4], raw[4], next[4];
    float mass[4];
    float freeMass=0.f;
    NMFEMNodeStateGPU nodes[4];
    privateCount=0u;
    for(uint k=0;k<4;++k){
        const uint n=t.nodes[k];
        nodes[k]=candidate[n];
        mass[k]=nodes[k].positionAndMass.w;
        owned[k]=owned[k] && nodes[k].restAndFixed.w<=.5f && mass[k]>0.f;
        if(owned[k]){freeMass+=mass[k];++privateCount;}
        raw[k]=direction[n].xyz;
        next[k]=raw[k];
        if(k)gRef[k]=transpose(inv)[k-1];
    }
    if(privateCount<2u || !(freeMass>0.f))return;
    gRef[0]=-gRef[1]-gRef[2]-gRef[3];
    const float3x3 start=registeredTrialF(nodes,inv,reference);
    const float j0=determinant(start);
    if(!(j0>=lo && j0<=hi))return;
    const float3x3 cof0=cofactorMatrix(start);
    float last=INFINITY;
    for(uint pass=0u;pass<12u;++pass){
        for(uint k=0;k<4;++k){
            nodes[k]=femTrialNodeAtAlpha(accepted[t.nodes[k]],candidate[t.nodes[k]],next[k],1.f,dt,true);
        }
        const float3x3 finish=registeredTrialF(nodes,inv,reference);
        const float3x3 delta=finish-start,cof1=cofactorMatrix(finish);
        const float j3=determinant(finish);
        // Mixed determinants are the Bernstein coefficients of det(F0 + alpha*dF).
        // Their convex hull bounds the polynomial on the full physical trial segment.
        float j1=j0,j2=j3;
        for(uint axis=0;axis<3;++axis){
            j1+=dot(cof0[axis],delta[axis])/3.f;
            j2-=dot(cof1[axis],delta[axis])/3.f;
        }
        const float3 value=float3(j1,j2,j3);
        if(!all(isfinite(value)))return;
        const float positiveScale=max(max(abs(j0),abs(j3)),hi);
        const float guard=min(.125f*(hi-lo),32.f*0x1p-23f*positiveScale);
        const float lower=lo+guard,upper=hi-guard;
        const float3 error=clamp(value,float3(lower),float3(upper))-value;
        uint limiting=0u;
        for(uint a=1u;a<3u;++a)if(abs(error[a])>abs(error[limiting]))limiting=a;
        const float violation=abs(error[limiting]);
        if(violation==0.f || (pass>0u && all(value>lo) && all(value<hi))){
            if(pass)for(uint k=0u;k<4u;++k)if(owned[k])direction[t.nodes[k]].xyz=next[k];
            return;
        }
        if(!(violation<last))return;
        last=violation;
        float3x3 derivative=cof1;
        if(limiting==0u)derivative=cof0*(1.f/3.f);
        if(limiting==1u)derivative=cofactorDirectional(start,finish)*(1.f/3.f);
        float3 g[4],sum=float3(0);
        for(uint k=0;k<4;++k){
            g[k]=owned[k]?derivative*gRef[k]:float3(0);
            sum+=g[k];
        }
        float denominator=0.f;
        for(uint k=0;k<4;++k)if(owned[k]){
            g[k]-=sum*(mass[k]/freeMass);
            denominator+=dot(g[k],g[k])/mass[k];
        }
        if(!(denominator>0.f) || !isfinite(denominator))return;
        const float lambda=error[limiting]/(dt*denominator);
        for(uint k=0;k<4;++k)if(owned[k])next[k]+=lambda*g[k]/mass[k];
        // Undo only momentum roundoff in the emitted correction. Then re-evaluate the path.
        for(uint axis=0;axis<3;++axis){
            float momentum=0.f,minResolution=INFINITY;uint pick=0u;
            for(uint k=0;k<4;++k)if(owned[k]){
                momentum=fma(mass[k],next[k][axis]-raw[k][axis],momentum);
                const float resolution=mass[k]*max(abs(next[k][axis]),1.e-30f);
                if(resolution<minResolution){minResolution=resolution;pick=k;}
            }
            next[pick][axis]-=momentum/mass[pick];
        }
    }'''
    (dst/'metal/fem.metalinc').write_text(s[:begin]+code+s[end:])
    prior=json.loads((base/'build.json').read_bytes())
    cmds=[[v.replace(str(base/'metal')+'/',str(out/'metal')+'/') for v in c] for c in prior['commands'][:2]]
    for cmd in cmds:subprocess.run(cmd,check=True)
    (out/'build.json').write_text(json.dumps(dict(commands=cmds,baseline_metal=digest(base/'metal/NumiMatter.metallib'),metal=digest(out/'metal/NumiMatter.metallib'),parent_source=digest(path),source=digest(dst/'metal/fem.metalinc')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
