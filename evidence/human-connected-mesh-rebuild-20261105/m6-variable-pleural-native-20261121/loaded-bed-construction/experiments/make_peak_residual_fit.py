"""At otherwise clear geometry, fit unilateral tractions to the worst nodal residuals."""
from pathlib import Path
import shutil
src=Path('initial-tractions-owner');dst=Path('peak-traction-owner');dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/'pelvic_contact.metalinc';s=p.read_text()
a=s.index('kernel void nm_pelvic_fit_tractions');b=s.index('kernel void nm_pelvic_residual_displacement',a)
k=s[a:b]
k=k.replace('device float4* residual [[buffer(9)]], uint side','device float4* residual [[buffer(9)]], constant uint& peak [[buffer(10)]], device const float4* quality [[buffer(11)]], uint side')
k=k.replace('    if (!needed[side]) return;','''    if (!needed[side]) return;
    if(peak){
        float4 q=quality[side];
        if(!all(isfinite(q)) || q.x<=.01f*max(q.y,1.e-6f) || q.z>=5.e-7f || q.w>2.e-9f) return;
    }''')
old='''            float gradient = 0;
            for (uint c = 0; c < 4; ++c) {
                if (site.weights[c] != 0 && freeRows[site.nodes[c]].z)
                    gradient += site.weights[c] * dot(normals[i].xyz, residual[site.nodes[c]].xyz);
            }
            float next = max(0.f, pressure[i] + gradient / diagonal[j]);
            float delta = next - pressure[i];
            float gain = delta * (2 * gradient - diagonal[j] * delta);'''
new='''            float gradient = 0, curvature = 0;
            for (uint c = 0; c < 4; ++c) {
                float w=site.weights[c];
                if(w==0 || !freeRows[site.nodes[c]].z)continue;
                float3 r=residual[site.nodes[c]].xyz, n=normals[i].xyz;
                float rn=dot(n,r);
                if(peak){
                    float rr=dot(r,r);
                    gradient+=w*rn*rr;
                    curvature+=w*w*(dot(n,n)*rr+2.f*rn*rn);
                }else gradient+=w*rn;
            }
            float h=peak?curvature:diagonal[j];
            if(!(h>0))continue;
            float next = max(0.f, pressure[i] + gradient / h);
            float delta = next - pressure[i];
            float gain = delta * (2 * gradient - diagonal[j] * delta);
            if(peak){
                // Exact decrease in sum ||r||^4. A step must improve the
                // fixed-geometry objective before it can compete for admission.
                gain=0;
                for(uint c=0;c<4;++c){
                    float w=site.weights[c];if(w==0 || !freeRows[site.nodes[c]].z)continue;
                    float3 r=residual[site.nodes[c]].xyz,n=normals[i].xyz;
                    float z=w*delta,change=z*(z*dot(n,n)-2.f*dot(r,n));
                    gain-=change*(2.f*dot(r,r)+change);
                }
            }'''
assert k.count(old)==1;k=k.replace(old,new);p.write_text(s[:a]+k+s[b:])
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text();old='bind(9,tractionResidual);run(2,32);';assert s.count(old)==1
s=s.replace(old,'bind(9,tractionResidual);words(10,{0,0,0,0});bind(11,quality);run(2,32);')
mark='        [e setComputePipelineState:gate];words(0,{p.stepIndex,0,0,unsigned(post)});';assert s.count(mark)==1
s=s.replace(mark,'''        if constexpr(pelvicMotionRegistry){
            unsigned count=unsigned(spanIndex.size());
            [e setComputePipelineState:makeSpans];[e setBytes:&count length:4 atIndex:0];
            bind(1,needed);bind(2,dispatchSizes);bind(3,dispatchArgs);run((count+31)/32,32);optional=true;
            [e setComputePipelineState:fitTractions];bind(0,needed);bind(1,poolSlots);bind(2,pool);
            bind(3,localN);bind(4,rows);bind(5,kr);bind(6,bulkGrad);bind(7,pressures);bind(8,inc);
            bind(9,tractionResidual);words(10,{1,0,0,0});bind(11,quality);run(2,32);
            [e setComputePipelineState:candidatePressure];words(0,{1,0,0,0});bind(1,poolSlots);
            bind(2,rows);bind(3,gram);bind(4,pressures);bind(5,stats);bind(6,needed);run(2,32);
            certify(48);optional=false;
        }
'''+mark);p.write_text(s)
