"""Use the fresh registered gap for the Schur direction and admit newly discovered contacts."""
import shutil,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:]);dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
m=dst/'pelvic_contact.metalinc';s=m.read_text()
old='rows[i].normal.w=rows[i].leverB.w-dot(normals[i].xyz,h)-prior;'
if s.count(old)!=1:raise ValueError('Missing mixed direction')
s=s.replace(old,'''// nearest(0) supplied the existing registered search offset. The
    // physical force and clearance certificates still use the raw observed gap.
    rows[i].normal.w-=dot(normals[i].xyz,h)+prior;''')
s+='''
kernel void nm_pelvic_mark_rescue(constant uint4& info [[buffer(0)]],device const float4* quality [[buffer(1)]],
    device const uint* needed [[buffer(2)]],device float4* log [[buffer(3)]],uint side [[thread_position_in_grid]]){
    if(side>=2 || info.x>=info.y)return;
    uint base=info.x*18+side;
    if(info.z==0){
        if(!needed[side])return;
    }else if(all(log[base]==0))return;
    log[base+2*info.z]=quality[side];
}
'''
m.write_text(s)
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
def one(a,b):
    global s
    if s.count(a)!=1:raise ValueError(a)
    s=s.replace(a,b)
one('rescueProposal=nil;', 'rescueProposal=nil,markRescue=nil;\n    id<MTLBuffer> rescueHistory=nil;')
one('            rescueProposal=pipeline(@"nm_pelvic_rescue_proposal");','''            rescueProposal=pipeline(@"nm_pelvic_rescue_proposal");
            markRescue=pipeline(@"nm_pelvic_mark_rescue");
            rescueHistory=alloc(std::max(1u,countTrace)*18u*16u,true);
            std::memset(rescueHistory.contents,0,rescueHistory.length);''')
a='''                for(unsigned attempt=0;attempt<8;++attempt){'''
one(a,'''                auto recordRescue=[&](unsigned index){
                    if(!countTrace)return;
                    [e setComputePipelineState:markRescue];words(0,{p.stepIndex,countTrace,index,0});
                    bind(1,quality);bind(2,needed);bind(3,rescueHistory);run(1,32);
                };
                recordRescue(0);
'''+a)
one('                    [e setComputePipelineState:rescueResidual];','''                    // A previously inactive neighboring face can become load bearing.
                    // Exchange only certified empty sites, then rebuild the incidence
                    // of all force-carrying points before constructing the direction.
                    select(1,false);nearest(0,1);compliance();
                    [e setComputePipelineState:clearInc];[e setBytes:&countNodes length:4 atIndex:0];bind(1,inc);run((countNodes*8+127)/128,128);
                    [e setComputePipelineState:buildInc];bind(0,poolSlots);bind(1,pressures);bind(2,pool);bind(3,inc);run(16,128);
                    [e setComputePipelineState:rescueResidual];''')
one('                    certify(128+attempt);optional=false;', '                    certify(128+attempt);recordRescue(1+attempt);optional=false;')
one('        std::ofstream physical(','''        if(rescueHistory){
            auto z=static_cast<const mr_float4*>(rescueHistory.contents);
            std::ofstream study(std::filesystem::path(file).parent_path()/"registered-equilibrium-iterations.csv");
            study<<"input_step,side,iteration,free_residual_N,external_force_N,release_gap_m,penetration_m\\n"<<std::setprecision(10);
            for(unsigned k=0;k<countTrace;++k)for(unsigned j=0;j<9;++j)for(unsigned b=0;b<2;++b){
                auto v=z[18*k+2*j+b];if(v.x==0 && v.y==0 && v.z==0 && v.w==0)continue;
                study<<k<<','<<b<<','<<j<<','<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<'\\n';
            }
        }
        std::ofstream physical(''')
p.write_text(s)
