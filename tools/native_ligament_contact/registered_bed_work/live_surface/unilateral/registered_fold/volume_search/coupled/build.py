"""Allow simultaneous volume proposals at adjacent cells without moving their shared search entries."""
from pathlib import Path
import json,subprocess,sys
from material_context import sha
def read(p):return json.loads(p.read_text())
def sub(s,k,v):
    if s.count(k)!=1:raise ValueError(('unexpected source site',k[:65],s.count(k)))
    return s.replace(k,v)
def build(case):
    old=case/'dynamic';exp=case/'coupled';src=old/'build';out=exp/'build';out.mkdir(exist_ok=True)
    log=old/'three/stderr.log'
    failure=read(old/'three/failure.json')
    if failure['segment_start']!=2:raise ValueError('Expected the measured shared-interface stall')
    if not (out/'src').exists():subprocess.run(['cp','-cR',src/'src',out/'src'],check=True)
    name='src/metal/fem.metalinc'
    s=(src/name).read_text();start=s.index('kernel void nm_fem_registered_volume_hints(')
    prefix=s[:start];s=s[start:]
    s=sub(s,'atomic_store_explicit(&claims[id[k]],0xffffffffu','atomic_store_explicit(&claims[id[k]],0u')
    s=sub(s,'atomic_fetch_min_explicit(&claims[id[k]],e,','atomic_fetch_add_explicit(&claims[id[k]],1u,')
    k='''    for(uint k=0;k<4;++k)
        if(atomic_load_explicit(&claims[t.nodes[k]],memory_order_relaxed)!=rows[row])return;'''
    s=sub(s,k,'''    bool owned[4];uint privateCount=0;
    for(uint k=0;k<4;++k){
        owned[k]=atomic_load_explicit(&claims[t.nodes[k]],memory_order_relaxed)==1u;
        privateCount+=uint(owned[k]);
    }
    if(privateCount<2u)return;''')
    # The common point keeps its original Newton motion in every row. Only the
    # additional feasibility adjustment is restricted to private coordinates.
    s=sub(s,'float momentum=0.f,finest=INFINITY;uint pick=0;', 'float momentum=0.f,finest=INFINITY;uint pick=0;')
    s=sub(s,'                    const float mass=candidate[t.nodes[k]].positionAndMass.w;','                    if(!owned[k])continue;\n                    const float mass=candidate[t.nodes[k]].positionAndMass.w;')
    s=sub(s,'for(uint k=0;k<4;++k)direction[t.nodes[k]].xyz=next[k];','for(uint k=0;k<4;++k)if(owned[k])direction[t.nodes[k]].xyz=next[k];')
    k='''        for(uint k=0;k<4;++k){
            gradient[k]=cofactor*gRef[k];
            denominator+=w[k]*dot(gradient[k],gradient[k]);
        }'''
    s=sub(s,k,'''        float3 sumGradient=0.f;float freeMass=0.f;
        for(uint k=0;k<4;++k){
            gradient[k]=owned[k]?cofactor*gRef[k]:float3(0);
            sumGradient+=gradient[k];
            if(owned[k])freeMass+=candidate[t.nodes[k]].positionAndMass.w;
        }
        for(uint k=0;k<4;++k)if(owned[k]){
            // Mass-orthogonal projection onto zero resultant on private nodes.
            gradient[k]-=(candidate[t.nodes[k]].positionAndMass.w/freeMass)*sumGradient;
            denominator+=w[k]*dot(gradient[k],gradient[k]);
        }''')
    dst=out/name;dst.unlink(missing_ok=True);dst.write_text(prefix+s)
    parent=read(src/'receipt.json');commands=[[v.replace(str(src)+'/',str(out)+'/') for v in c] for c in parent['commands'][:2]]
    for c in commands:subprocess.run(c,check=True)
    l=read(old/'parent/launch.json');env=l['environment'];key='NUMI_MATTER_METALLIB_OVERRIDE'
    if l['pins'].pop(env[key])!=parent['metallib']:raise ValueError('Wrong native library for shared directions')
    env[key]=str(out/'NumiMatter.metallib');l['pins'][env[key]]=sha(out/'NumiMatter.metallib')
    (exp/'parent').mkdir(exist_ok=True);(exp/'parent/launch.json').write_text(json.dumps(l,indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps(dict(parent=sha(src/'receipt.json'),source=sha(dst),failure=sha(log),
        metallib=sha(out/'NumiMatter.metallib'),commands=commands),indent=2)+'\n')
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
