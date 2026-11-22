#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <array>
#include <cstring>
#include <vector>
#include <cstdio>

int main(int argc,const char** argv) {
  @autoreleasepool {
    if(argc!=2)return 2;
    auto gpu=MTLCreateSystemDefaultDevice();NSError* error=nil;
    auto lib=[gpu newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    auto pipe=[&](NSString* name){
        auto f=[lib newFunctionWithName:[@"numi_matter_metal::" stringByAppendingString:name]];
        return f ? [gpu newComputePipelineStateWithFunction:f error:&error] : nil;
    };
    auto hint=pipe(@"nm_fem_registered_volume_hints"),claim=pipe(@"nm_fem_registered_volume_claim");
    if(!hint||!claim){NSLog(@"%@",error);return 3;}
    auto buf=[&](size_t n){return [gpu newBufferWithLength:n options:MTLResourceStorageModeShared];};
    constexpr unsigned tiles=3,cells=3*256,owner=0;
    auto tets=buf(cells*sizeof(NMTetrahedronGPU)),partials=buf(tiles*sizeof(nm_float4));
    auto rows=buf(tiles*sizeof(unsigned)),claims=buf(4*cells*sizeof(unsigned));
    auto* t=static_cast<NMTetrahedronGPU*>(tets.contents);
    for(unsigned i=0;i<cells;++i)t[i].nodes={4*i,4*i+1,4*i+2,4*i+3};
    t[300].nodes.x=t[20].nodes.x;
    NMContinuumObjectGPU object{};object.elementCount=cells;
    NMMatterDispatchGPU d{};d.environmentCount=d.objectCount=1;d.femNodeCount=4*cells;
    auto queue=[gpu newCommandQueue];
    auto dispatch=[&](unsigned it){
        auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:hint];[enc setBytes:&d length:sizeof(d) atIndex:0];
        [enc setBytes:&object length:sizeof(object) atIndex:1];[enc setBuffer:partials offset:0 atIndex:2];
        [enc setBuffer:rows offset:0 atIndex:3];[enc setBuffer:claims offset:0 atIndex:4];[enc setBuffer:tets offset:0 atIndex:5];
        [enc setBytes:&owner length:4 atIndex:6];[enc setBytes:&tiles length:4 atIndex:7];[enc setBytes:&it length:4 atIndex:8];
        [enc dispatchThreads:MTLSizeMake(tiles,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:claim];[enc setBuffer:rows offset:0 atIndex:0];[enc setBuffer:claims offset:0 atIndex:1];
        [enc setBuffer:tets offset:0 atIndex:2];[enc setBytes:&tiles length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(tiles,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];return cb.status==MTLCommandBufferStatusCompleted;
    };
    auto* p=static_cast<nm_float4*>(partials.contents);
    auto* ids=static_cast<unsigned*>(rows.contents);auto* byNode=static_cast<unsigned*>(claims.contents);
    constexpr unsigned invalid=0xffffffffu;
    auto testRows=[&](std::array<unsigned,3> want){return std::memcmp(ids,want.data(),12)==0;};
    auto fail=[&](unsigned tile,unsigned id,float reason){std::memcpy(&p[tile].x,&id,4);p[tile].w=reason;};
    fail(0,20,2);fail(1,300,4);fail(2,604,2);
    if(!dispatch(0)||!testRows({invalid,invalid,invalid}))return 4;
    if(!dispatch(1)||!testRows({20,300,604})||byNode[t[300].nodes.x]!=20||byNode[t[604].nodes.x]!=604)return 5;
    fail(0,21,4);fail(1,42,2);fail(2,605,8);
    if(!dispatch(2)||!testRows({21,300,604})||byNode[t[300].nodes.x]!=300||byNode[t[21].nodes.y]!=21)return 6;
    if(!dispatch(0)||!testRows({invalid,invalid,invalid}))return 7;
    std::printf("device=%s conflict_claim=pass stale_epoch_reset=pass retained_hint=pass out_of_tile=pass constitutive_only=pass\n",gpu.name.UTF8String);
  }
}
