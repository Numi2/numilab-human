"""Read selected solver intermediates on one root without changing any physical buffer."""
from pathlib import Path
import shutil, sys
src,dst=map(Path,sys.argv[1:]);dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
kernel="""
kernel void nm_pelvic_audit_stage(constant uint4& d [[buffer(0)]],
    device const float4* u [[buffer(1)]], device const float4* shift [[buffer(2)]],
    device const uint* slots [[buffer(3)]],device const float* pressure [[buffer(4)]],
    device const PelvicPoolSample* pool [[buffer(5)]], device const uint* lookup [[buffer(6)]],
    device const PelvicRow* rows [[buffer(7)]],device const float* matrix [[buffer(8)]],
    device atomic_uint* out [[buffer(9)]],uint i [[thread_position_in_grid]]){
    uint side=i/65536,j=i%65536;
    if(side>1)return;
    device atomic_uint* s=out+12*(2*d.x+side);
    if(j<pelvicFlex[side].y){
        uint n=pelvicFlex[side].x+j;
        float3 a=u[n].xyz, b=shift[n].xyz;
        if(!all(isfinite(a)))atomic_fetch_add_explicit(s+0,1u,memory_order_relaxed);
        else atomic_fetch_max_explicit(s+7,as_type<uint>(length(a)),memory_order_relaxed);
        if(!all(isfinite(b)))atomic_fetch_add_explicit(s+1,1u,memory_order_relaxed);
    }
    uint base=256*side, row=base+j/256, col=base+j%256;
    uint query=slots[col];if(query==UINT_MAX)return;
    if(j<256){
        auto p=pool[query];float f=pressure[col],g=rows[col].normal.w;
        if(!isfinite(f))atomic_fetch_add_explicit(s+2,1u,memory_order_relaxed);
        else atomic_fetch_max_explicit(s+8,as_type<uint>(abs(f)),memory_order_relaxed);
        for(uint k=0;k<4;++k)if(p.weights[k]!=0.f && lookup[p.nodes[k]]==UINT_MAX){
            atomic_fetch_add_explicit(s+3,1u,memory_order_relaxed);
            atomic_fetch_max_explicit(s+10,query,memory_order_relaxed);
        }
        if(!isfinite(g)){
            atomic_fetch_add_explicit(s+4,1u,memory_order_relaxed);
            atomic_fetch_max_explicit(s+11,query,memory_order_relaxed);
        }
    }
    if(slots[row]!=UINT_MAX){
        float m=matrix[i];
        if(!isfinite(m))atomic_fetch_add_explicit(s+5,1u,memory_order_relaxed);
        else atomic_fetch_max_explicit(s+9,as_type<uint>(abs(m)),memory_order_relaxed);
    }
}
"""
p=dst/"pelvic_contact.metalinc";p.write_text(p.read_text()+kernel)
p=dst/"NumiHumanPelvicContact.hpp";s=p.read_text()
def before(needle,text):
    global s
    assert s.count(needle)==1,needle
    s=s.replace(needle,text+needle)
before('    unsigned observedForceRoot',"""    id<MTLBuffer> stageAudit=nil;
    id<MTLComputePipelineState> stagePipeline=nil;
    std::vector<std::string> stageNames;
""")
before('                forceNodes=alloc',"""                stageAudit=alloc(1024*2*12*4,true);std::memset(stageAudit.contents,0,stageAudit.length);
                stagePipeline=pipeline(@"nm_pelvic_audit_stage");
""")
before('        auto nearest=',"""        auto auditStage=[&](const char* name){
            if(p.stepIndex!=observedForceRoot || !stageAudit)return;
            if(stageNames.size()>=1024)throw std::runtime_error("contact stage audit capacity");
            [e setComputePipelineState:stagePipeline];words(0,{unsigned(stageNames.size()),0,0,0});
            stageNames.push_back(std::string(post?"post/":"pre/")+name);
            bind(1,coordinates);bind(2,baseShift);bind(3,poolSlots);bind(4,pressures);bind(5,pool);
            bind(6,freeIndex);bind(7,rows);bind(8,gram);bind(9,stageAudit);
            [e dispatchThreadgroups:MTLSizeMake(1024,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
""")
start=s.index("        bulkForces();\n        // A pressure")
end=s.index("        if(post && p.stepIndex==observedForceRoot",start)
sub=s[start:end]
sub=sub.replace("certify(UINT_MAX);",'certify(UINT_MAX); auditStage("initial");')
for name,label in (("factorApply","factor"),("solve","pre_schur"),("moveNodes","post_schur"),("dualMove","post_schur"),("nodeEquilibrium","pre_free_material")):
    key="[e setComputePipelineState:"+name+"];"
    sub=sub.replace(key,'auditStage("'+label+'");'+key)
sub=sub.replace("observe(1,1);",'auditStage("before_surface_update");observe(1,1);')
sub=sub.replace("certify(outer);",'certify(outer);auditStage("mechanical_certificate");')
s=s[:start]+sub+s[end:]
before('            auto emit=[&]',"""            auto result=static_cast<const uint32_t*>(stageAudit.contents);
            std::ofstream t(dir/"contact-stage-audit.csv");
            t<<"index,stage,side,bad_coordinate,bad_shift,bad_pressure,missing_parent,bad_gap,bad_gram,reserved,max_u_m,max_pressure_N,max_gram,missing_query,bad_gap_query\\n";
            for(size_t i=0;i<stageNames.size();++i)for(unsigned side=0;side<2;++side){
                auto v=result+12*(2*i+side);t<<i<<','<<stageNames[i]<<','<<side;
                for(unsigned k=0;k<12;++k){t<<',';if(k>=7 && k<=9){float f;std::memcpy(&f,&v[k],4);t<<f;}else t<<v[k];}
                t<<'\\n';
            }
""")
p.write_text(s)
