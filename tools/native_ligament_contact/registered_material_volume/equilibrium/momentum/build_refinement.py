"""Continue the direction solve against the represented, momentum-balanced configuration."""
import json,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest

def run(base,out):
    src=base/'metal/src';dst=out/'metal/src';(dst/'metal').mkdir(parents=True,exist_ok=True)
    for p in [*src.iterdir(),*(src/'metal').iterdir()]:
        if p.is_file() and p.name!='fem.metalinc':
            target=dst/p.relative_to(src)
            if not target.exists():target.symlink_to(p.resolve())
    ref=src/'metal/fem.metalinc'
    s=ref.read_text();i=s.index('inline uint registeredVolumeCellDirection');j=s.index('kernel void ',i);b=s[i:j]
    a='''            if(!(actual>lo && actual<hi))return 8u;

            for(uint k=0;k<4;++k)if(owned[k])direction[t.nodes[k]].xyz=next[k];
            return 6u;'''
    replacement='''            if(actual>lo && actual<hi){
                for(uint k=0;k<4;++k)if(owned[k])direction[t.nodes[k]].xyz=next[k];
                return refinementsAfterBalance?11u:6u;
            }
            if(privateCount!=4u || !isfinite(actual))return 8u;
            // The discrete momentum correction is part of the represented
            // direction. Rebuild its gradient at the state that actually failed.
            // The material interval, acceptance certificate and force residual
            // remain the authority for every trial.
            ++refinementsAfterBalance;
            J=actual;
            F=registeredTrialF(p,inv,reference);
            const float activeLimit=J<=lo?lo:hi;
            const float margin=min(.25f*(hi-lo),max(abs(target-activeLimit),
                2.f*abs(unbalancedJ-actual)+32.f*0x1p-23f*max(1.f,abs(activeLimit))));
            target=J<=lo?lo+margin:hi-margin;
            if(!(target>lo && target<hi))return 8u;
            lastError=INFINITY;'''
    if b.count(a)!=1:raise ValueError('Wrong projection owner')
    b=b.replace(a,replacement)
    b=b.replace('for(uint refine=0;refine<=5;++refine)', 'uint refinementsAfterBalance=0u;\n    for(uint refine=0;refine<=5u+min(refinementsAfterBalance,3u);++refine)')
    b=b.replace('const float3x3 F=registeredTrialF(p,inv,reference);','float3x3 F=registeredTrialF(p,inv,reference);').replace('const float J=determinant(F);','float J=determinant(F);const float unbalancedJ=J;')
    (dst/'metal/fem.metalinc').write_text(s[:i]+b+s[j:])
    d=json.loads((base/'build.json').read_bytes())
    if digest(base/'metal/NumiMatter.metallib')!=d['shader']:raise ValueError('Mismatched parent')
    cmds=[[v.replace(str(base/'metal')+'/',str(out/'metal')+'/') for v in q] for q in d['commands']]
    with (out/'compile.log').open('w') as f:
        for q in cmds:subprocess.run(q,stdout=f,stderr=subprocess.STDOUT,check=True)
    (out/'build.json').write_text(json.dumps(dict(parent_shader=d['shader'],source=digest(dst/'metal/fem.metalinc'),original_source=digest(ref),shader=digest(out/'metal/NumiMatter.metallib'),commands=cmds),indent=2)+'\n')
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
