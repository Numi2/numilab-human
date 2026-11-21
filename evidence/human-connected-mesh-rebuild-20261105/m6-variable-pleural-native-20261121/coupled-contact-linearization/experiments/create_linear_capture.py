from pathlib import Path
import shutil
base=Path('motion-registry-owner');out=Path('linear-observation-owner');out.mkdir(exist_ok=True)
for f in base.iterdir():
    if f.is_file():shutil.copyfile(f,out/f.name)
p=out/'pelvic_contact.metalinc';s=p.read_text();s+='''
// A bounded observer of the actual linearized left contact problem. It is
// scheduled only for the requested root, following the owning Schur dispatch.
kernel void nm_pelvic_capture_linear(constant uint4& d [[buffer(0)]],device const uint* slots [[buffer(1)]],
    device const PelvicRow* rows [[buffer(2)]],device const float* gram [[buffer(3)]],
    device const float* incoming [[buffer(4)]],device const float* out [[buffer(5)]],
    device const uint* needed [[buffer(6)]],device const float4* metrics [[buffer(7)]],
    device const uint* material [[buffer(8)]],device float* record [[buffer(9)]],
    uint i [[thread_position_in_grid]]){
    device float* frame=record+d.x*67600u;uint side=d.y,base=side*256;
    if(i<65536)frame[i]=gram[65536*side+i];
    if(i<256){
        frame[65536+i]=as_type<float>(slots[base+i]);frame[65792+i]=rows[base+i].normal.w;
        frame[66048+i]=incoming[base+i];frame[66304+i]=out[base+i];
        frame[66560+i]=rows[base+i].leverB.w;
    }
    if(!i){frame[66816]=float(needed[side]);frame[66817]=d.z?float(material[side]):-1.f;
        for(uint k=0;k<4;++k)frame[66818+k]=metrics[side][k];}
}
'''
p.write_text(s)
p=out/'NumiHumanPelvicContact.hpp';s=p.read_text()
s=s.replace('id<MTLBuffer> liveExtras=nil,','id<MTLBuffer> linearEvidence=nil; id<MTLComputePipelineState> captureLinear=nil; unsigned captureRoot=UINT_MAX;\n    id<MTLBuffer> liveExtras=nil,')
s=s.replace('if constexpr(pelvicMotionRegistry){\n            motionOwners=', '''if constexpr(pelvicMotionRegistry){
            if(const char* step=std::getenv("NUMI_HUMAN_PELVIC_OBSERVE_LINEAR_ROOT")){
                captureRoot=unsigned(std::stoul(step));linearEvidence=alloc(32*67600*sizeof(float),true);
                std::memset(linearEvidence.contents,0,linearEvidence.length);
                captureLinear=pipeline(@"nm_pelvic_capture_linear");
            }
            motionOwners=''')
a='        for(unsigned outer=0;outer<16;++outer){'
s=s.replace(a,'''        auto observeLinear=[&](unsigned iteration){
            if(!post || p.stepIndex!=captureRoot || !linearEvidence)return;
            [e setComputePipelineState:captureLinear];words(0,{iteration,1,unsigned(iteration>=16),0});
            bind(1,poolSlots);bind(2,rows);bind(3,gram);bind(4,priorPressure);bind(5,pressures);bind(6,needed);bind(7,quality);bind(8,materialNeeded);bind(9,linearEvidence);
            [e dispatchThreadgroups:MTLSizeMake(512,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
'''+a)
a='            [e setComputePipelineState:moveNodes];';assert s.count(a)==1;s=s.replace(a,'            observeLinear(outer);\n'+a)
a='                [e setComputePipelineState:dualMove];';assert s.count(a)==1;s=s.replace(a,'                observeLinear(16+sweep);\n'+a)
a='        if constexpr(pelvicHasCrossingCertificate){\n            std::ofstream g';assert a in s
s=s.replace(a,'''        if(linearEvidence && encodedSteps>captureRoot){
            std::ofstream raw(std::filesystem::path(file).parent_path()/"observed-contact-linear.bin",std::ios::binary);
            raw.write(static_cast<const char*>(linearEvidence.contents),linearEvidence.length);
        }
'''+a)
p.write_text(s)
