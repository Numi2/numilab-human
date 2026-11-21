"""Install the full-cortex closest-point query ahead of the four native normal rows."""
import shutil,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:]);dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
metal=dst/'registered_digit_contact.metalinc';s=metal.read_text();i=s.index('kernel void nm_human_digit_cortical_rows(')
s=s[:i]+Path('digit_surface_query.metalinc').read_text()+'\n'+s[i:]
s=s.replace('device float4* observedPose [[buffer(12)]],','device float4* observedPose [[buffer(12)]],device const NMDigitHit* hits [[buffer(13)]],constant uint& stride [[buffer(14)]],')
i=s.index('    if(lane==0){\n        float3 delta=',s.index('kernel void nm_human_digit_cortical_rows'));j=s.index('        float3 solverNormal=',i)
s=s[:i]+'''    float minGap=INFINITY;uint best=UINT_MAX;
    for(uint k=lane;k<stride;k+=32){
        float g=hits[row*stride+k].a.w;
        if(g<minGap){minGap=g;best=k;}
    }
    float globalGap=simd_min(minGap);uint index=simd_min(minGap==globalGap?best:UINT_MAX);
    if(lane==0){
        auto h=hits[row*stride+index];
        NMDigitClosest c={h.a.xyz,h.b.xyz,h.n.xyz,h.a.w,float(index),float(as_type<uint>(h.b.w)),0};
        tmp[0]=h.a;tmp[1]=h.b;tmp[2]=h.n;
'''+s[j:]
# Surface witnesses replace the four original individual normals.
metal.write_text(s)
header=dst/'NumiHumanDigitCorticalContact.hpp';s=header.read_text()
def sub(a,b):
    global s
    if s.count(a)!=1:raise ValueError(a)
    s=s.replace(a,b)
sub('id<MTLComputePipelineState> pipeline=nil;', '''id<MTLComputePipelineState> pipeline=nil,facePipeline=nil;
    id<MTLBuffer> surface=nil,faceResults=nil;
    std::array<mr_uint4,4> surfaceRanges{};
    unsigned faceStride=0;''')
sub('[e setComputePipelineState:pipeline];','''[e setComputePipelineState:facePipeline];
        mr_uint4 faceSize{p.articulationFirstBody,faceStride,0,0};
        [e setBytes:&faceSize length:sizeof(faceSize) atIndex:0];
        [e setBytes:surfaceRanges.data() length:sizeof(surfaceRanges) atIndex:1];
        [e setBuffer:surface offset:0 atIndex:2];[e setBuffer:features offset:0 atIndex:3];
        [e setBuffer:(__bridge id<MTLBuffer>)p.bodyPoses offset:0 atIndex:4];
        [e setBuffer:(__bridge id<MTLBuffer>)p.bodyPositionLow offset:0 atIndex:5];
        [e setBuffer:faceResults offset:0 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(4*faceStride,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e memoryBarrierWithResources:&faceResults count:1];
        [e setComputePipelineState:pipeline];
        [e setBuffer:faceResults offset:0 atIndex:13];
        [e setBytes:&faceStride length:sizeof(faceStride) atIndex:14];'''.replace('[e memoryBarrierWithResources:&faceResults count:1];','id<MTLResource> dependency=faceResults;[e memoryBarrierWithResources:&dependency count:1];'))
sub('        features=[device newBufferWithBytes:', '''        auto query=[lib newFunctionWithName:@"nm_digit_surface_faces"];
        facePipeline=[device newComputePipelineStateWithFunction:query error:&error];
        require(facePipeline && facePipeline.threadExecutionWidth==32,"cortical surface query unavailable");
        const auto root=std::filesystem::path(std::getenv("NUMI_HUMAN_DIGIT_CORTICAL_FEATURES")).parent_path();
        std::ifstream geometry(root/"proximal-surface.bin",std::ios::binary|std::ios::ate);
        const auto length=geometry.tellg();
        require(length>std::streamoff(sizeof(surfaceRanges)) && length<4*1024*1024,"invalid cortical surface size");
        geometry.seekg(0);geometry.read(reinterpret_cast<char*>(surfaceRanges.data()),sizeof(surfaceRanges));
        std::vector<unsigned char> payload(std::size_t(length)-sizeof(surfaceRanges));
        require(bool(geometry.read(reinterpret_cast<char*>(payload.data()),payload.size())) && payload.size()%48==0,"invalid native cortical triangle data");
        for(auto bytes:std::as_bytes(std::span(surfaceRanges))){featureFingerprint^=std::to_integer<unsigned char>(bytes);featureFingerprint*=1099511628211ull;}
        for(auto bytes:payload){featureFingerprint^=bytes;featureFingerprint*=1099511628211ull;}
        for(auto range:surfaceRanges){
            require(range.y && range.w && std::uint64_t(range.x)+range.y<=payload.size()/48 &&
                    std::uint64_t(range.z)+range.w<=payload.size()/48,"cortical surface range exceeds its source");
            faceStride=std::max(faceStride,range.y);
        }
        surface=[device newBufferWithBytes:payload.data() length:payload.size() options:MTLResourceStorageModeShared];
        faceResults=[device newBufferWithLength:4*faceStride*48 options:MTLResourceStorageModePrivate];
        require(surface && faceResults,"could not allocate registered cortical surfaces");
        features=[device newBufferWithBytes:''')
header.write_text(s)
