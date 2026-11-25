#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

static NMDeformableContactGPU makeRow(nm_uint4 nodes, unsigned kind, unsigned source) {
    NMDeformableContactGPU c{};
    c.nodes=nodes;
    c.identity={11,13,kind,NM_CONTACT_VALID};
    c.source={source,42,source,1};
    c.normalAndTOI={1,0,0,0.5f};
    c.pointAndSeparation={0,0,0,1.e-6f};
    c.impulseAndFriction={2.f,-0.3f,0.2f,0.4f};
    c.barrier={2.f,3.f,1.e-5f,4.f};
    c.barrierHessianRow0={3.f,1.f,0.f,0.01f};
    c.barrierHessianRow1={1.f,2.f,0.f,0.f};
    c.barrierHessianRow2={0.f,0.f,1.f,0.000001f};
    c.mollifier={0.9f,0.3f,0.4f,4.f};
    return c;
}
static NMDeformableContactGPU suppress(NMDeformableContactGPU c) {
    c.impulseAndFriction={0,0,0,c.impulseAndFriction.w};
    c.barrier={0,0,c.barrier.z,0};
    c.barrierHessianRow0={0,0,0,c.barrierHessianRow0.w};
    c.barrierHessianRow1={0,0,0,c.barrierHessianRow1.w};
    c.barrierHessianRow2={0,0,0,c.barrierHessianRow2.w};
    c.mollifier={c.mollifier.x,0,0,0};return c;
}
int main(int argc,char**argv){@autoreleasepool {
    if(argc!=2)return 2;
    std::vector<NMDeformableContactGPU> rows={
        makeRow({3,4,6,7},1,18),makeRow({4,3,6,7},1,5),
        makeRow({3,4,7,6},1,20),makeRow({4,3,7,6},1,17),
        makeRow({5,9,10,11},0,3),makeRow({5,11,9,10},0,2),
        makeRow({5,10,11,9},0,7),makeRow({5,9,10,12},0,11),
        makeRow({3,4,6,7},1,1),makeRow({0,4,6,7},1,1),
        makeRow({0,4,7,6},1,2),makeRow({3,4,6,7},1,1),
        makeRow({3,4,6,7},1,18),makeRow({3,4,6,7},1,5)
    };
    rows[8].identity.w|=NM_CONTACT_MAPPED_FEATURE;
    rows[11].source.w=2;
    rows[12].identity.y=rows[12].identity.x;
    rows[13].identity.x=13;rows[13].identity.y=11;
    auto expected=rows;
    for(unsigned i:{0u,2u,3u,4u,6u})expected[i]=suppress(expected[i]);
    constexpr unsigned nodes=16, envs=2;
    NMMatterDispatchGPU d{};d.environmentCount=envs;d.femNodeCount=nodes-1;
    d.gridNodeCount=1;d.objectCount=15;d.deformableContactCapacity=rows.size();
    std::vector<NMDeformableContactGPU> input=rows;
    auto reversed=rows;std::reverse(reversed.begin(),reversed.end());
    input.insert(input.end(),reversed.begin(),reversed.end());
    auto revExpected=expected;std::reverse(revExpected.begin(),revExpected.end());
    expected.insert(expected.end(),revExpected.begin(),revExpected.end());
    std::vector<NMIncidenceRangeGPU> ranges(nodes*envs);
    std::vector<unsigned> scatter;
    for(unsigned e=0;e<envs;e++)for(unsigned n=0;n<nodes;n++) {
        auto& r=ranges[e*nodes+n];r.first=scatter.size();
        for(unsigned j=0;j<rows.size();j++){
            auto s=input[e*rows.size()+j].nodes;
            std::array<unsigned,4> v={s.x,s.y,s.z,s.w};
            for(unsigned k=0;k<4;k++)if(v[k]==n)scatter.push_back((j<<3)|k);
        }
        r.count=scatter.size()-r.first;
    }
    NSError* err=nil;auto device=MTLCreateSystemDefaultDevice();
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
    auto function=[library newFunctionWithName:@"numi_matter_metal::nm_contact_own_shared_edge_forces"];
    auto pipeline=[device newComputePipelineStateWithFunction:function error:&err];
    if(!pipeline){std::fprintf(stderr,"pipeline error %s\n",err.localizedDescription.UTF8String);return 1;}
    auto contacts=[device newBufferWithBytes:input.data() length:input.size()*sizeof(input[0]) options:MTLResourceStorageModeShared];
    auto incidence=[device newBufferWithBytes:scatter.data() length:scatter.size()*sizeof(scatter[0]) options:MTLResourceStorageModeShared];
    auto owners=[device newBufferWithBytes:ranges.data() length:ranges.size()*sizeof(ranges[0]) options:MTLResourceStorageModeShared];
    auto queue=[device newCommandQueue];
    // Re-execution on a converged candidate must be idempotent.
    for(unsigned pass=0;pass<2;pass++){
        auto cb=[queue commandBuffer]; auto enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];[enc setBytes:&d length:sizeof(d) atIndex:0];
        [enc setBuffer:contacts offset:0 atIndex:1];[enc setBuffer:incidence offset:0 atIndex:2];
        [enc setBuffer:owners offset:0 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(nodes*envs,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted)return 1;
        for(size_t i=0;i<input.size();++i)
            if(std::memcmp(static_cast<NMDeformableContactGPU*>(contacts.contents)+i,&expected[i],sizeof(expected[i]))!=0){
                std::fprintf(stderr,"contact %zu differs on pass %u\n",i,pass);return 1;
            }
    }
    std::printf("%s: two environments, 28 records, reversed slot order, two passes, byte-exact geometry and expected constitutive terms\n",device.name.UTF8String);
    return 0;
}}
