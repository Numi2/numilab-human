"""Capture only the material frames that generate registered contact rows on the GPU."""
from pathlib import Path
import shutil,sys
src,dst=map(Path,sys.argv[1:]);dst.mkdir(exist_ok=True)
for p in src.iterdir():
    if p.is_file():shutil.copyfile(p,dst/p.name)
def patch(p,replacements):
    s=p.read_text()
    for needle,new in replacements:
        if s.count(needle)!=1:raise ValueError(needle)
        s=s.replace(needle,new)
    p.write_text(s)
patch(dst/'registered_digit_contact.metalinc',[
('uint row [[threadgroup_position_in_grid]],uint lane', 'device float4* observedPose [[buffer(12)]],\n    uint row [[threadgroup_position_in_grid]],uint lane'),
('    if(lane==0){\n        float3 delta=', '''    // The accepted input of the physical step is read from the native owner.
    // Each side has two registered features on the same rigid pair.
    if(dim.w<rows.w && !rows.z && !(row&1u) && lane<2){
        uint bidx=lane?b:a;
        uint off=(4*dim.w+row+lane)*3;
        observedPose[off]=float4(poses[bidx].position.xyz,1);
        observedPose[off+1]=float4(low[bidx].xyz,0);
        observedPose[off+2]=poses[bidx].orientation;
    }
    if(lane==0){
        float3 delta=''')])
patch(dst/'NumiHumanDigitCorticalContact.hpp',[
('features=nil,trace=nil;', 'features=nil,trace=nil,poseTrace=nil;'),
('[e setBuffer:trace offset:0 atIndex:9];','[e setBuffer:poseTrace offset:0 atIndex:12];[e setBuffer:trace offset:0 atIndex:9];'),
('require(features&&trace,"could not allocate cortical scatter");','''poseTrace=[device newBufferWithLength:std::max<std::size_t>(16,std::size_t(traceSteps)*12*sizeof(mr_float4)) options:MTLResourceStorageModeShared];
        require(features&&trace&&poseTrace,"could not allocate cortical scatter");
        for(unsigned r=0;r<source.size();r+=2)
            require(source[r].bodies.x==source[r+1].bodies.x && source[r].bodies.y==source[r+1].bodies.y,
                    "digit pose trace requires adjacent registered rows to share a cortical pair");'''),
('        std::ofstream f(evidenceDir+"/digit-contact-geometry.csv");','''        if(poseTrace){
            std::ofstream poses(evidenceDir+"/digit-material-frames.csv");
            poses<<"input_step,body,x,y,z,low_x,low_y,low_z,qx,qy,qz,qw\\n";
            poses.precision(10);
            auto v=static_cast<const mr_float4*>(poseTrace.contents);
            for(unsigned step=0;step<traceSteps;++step)for(unsigned j=0;j<4;++j){
                auto t=v+3*(4*step+j);
                if(t[0].w!=1.f)continue;
                auto pair=source[j&~1u].bodies;
                poses<<step<<','<<((j&1u)?pair.y:pair.x);
                for(unsigned ix=0;ix<3;++ix)poses<<','<<t[ix].x<<','<<t[ix].y<<','<<t[ix].z;
                poses<<','<<t[2].w<<'\\n';
            }
        }
        std::ofstream f(evidenceDir+"/digit-contact-geometry.csv");''')])
