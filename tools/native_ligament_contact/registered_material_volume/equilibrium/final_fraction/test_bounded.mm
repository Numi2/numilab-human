#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc,const char** argv) {@autoreleasepool {
    if(argc!=2)return 2;
    auto gpu=MTLCreateSystemDefaultDevice();
    NSError* error=nil;
    auto lib=[gpu newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    if(!lib)return 3;
    auto pipeline=[&](NSString* suffix) {
        auto name=[@"numi_matter_metal::nm_fem_registered_release_bound_" stringByAppendingString:suffix];
        auto fn=[lib newFunctionWithName:name constantValues:[MTLFunctionConstantValues new] error:&error];
        return fn?[gpu newComputePipelineStateWithFunction:fn error:&error]:nil;
    };
    auto begin=pipeline(@"begin"),elements=pipeline(@"elements"),finish=pipeline(@"finish");
    if(!begin || !elements || !finish)return 4;
    auto queue=[gpu newCommandQueue];
    auto allocate=[&](size_t n){auto b=[gpu newBufferWithLength:n options:MTLResourceStorageModeShared];std::memset(b.contents,0,n);return b;};
    auto d=allocate(sizeof(NMMatterDispatchGPU));auto* dispatch=(NMMatterDispatchGPU*)d.contents;
    dispatch->environmentCount=1;dispatch->objectCount=1;dispatch->tetrahedronCount=1;dispatch->gravityAndTimestep.w=1.f;
    auto m=allocate(sizeof(NMMicrostepGPU));
    auto ob=allocate(sizeof(NMContinuumObjectGPU));auto* o=(NMContinuumObjectGPU*)ob.contents;
    o->flags=NM_OBJECT_ACTIVE|NM_OBJECT_MIXED_FEM;o->representation=NM_REPRESENTATION_FEM;o->elementCount=1;
    auto material=allocate(sizeof(NMMaterialGPU));((NMMaterialGPU*)material.contents)->validity={.35f,2.5f,0,0};
    auto nodes=allocate(4*sizeof(NMFEMNodeStateGPU));auto* p=(NMFEMNodeStateGPU*)nodes.contents;
    p[0].positionAndMass={0,0,0,1};p[1].positionAndMass={1,0,0,1};p[2].positionAndMass={0,1,0,1};p[3].positionAndMass={0,0,2.25f,1};
    auto solution=allocate(4*sizeof(nm_float4));auto* dir=(nm_float4*)solution.contents;dir[1].x=.8f;dir[2].y=-.4f;
    auto mesh=allocate(sizeof(NMTetrahedronGPU));auto* t=(NMTetrahedronGPU*)mesh.contents;t->nodes={0,1,2,3};t->identity.w=NM_OBJECT_ACTIVE;
    t->inverseRestRow0={1,0,0,0};t->inverseRestRow1={0,1,0,0};t->inverseRestRow2={0,0,1,0};
    auto clock=allocate(sizeof(NMSchedulerStateGPU));((NMSchedulerStateGPU*)clock.contents)->numerical.w=1;
    auto adaptive=allocate(sizeof(NMAdaptiveStateGPU));((NMAdaptiveStateGPU*)adaptive.contents)->activeRepresentation=NM_REPRESENTATION_FEM;
    auto floors=allocate(16),alpha=allocate(4),scratch=allocate(48),flags=allocate(sizeof(NMFGMRESStateGPU)),status=allocate(sizeof(NMMatterStatusGPU)),per=allocate(16),dummy=allocate(256);
    auto dispatchBlock=[&](uint32_t pass,uint32_t last,uint32_t budget){
        auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
        auto bind=[&](std::initializer_list<id<MTLBuffer>> bufs){unsigned i=0;for(auto b:bufs)[enc setBuffer:b offset:0 atIndex:i++];};
        auto one=[&](){[enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];};
        [enc setComputePipelineState:begin];bind({d,alpha,scratch});[enc setBytes:&pass length:4 atIndex:3];[enc setBuffer:flags offset:0 atIndex:4];[enc setBuffer:status offset:0 atIndex:5];one();
        [enc setComputePipelineState:elements];bind({d,m,ob,material,nodes,nodes,solution,mesh,clock,adaptive,floors,alpha,scratch,dummy,dummy,dummy});[enc setBytes:&budget length:4 atIndex:16];
        [enc dispatchThreadgroupsWithIndirectBuffer:scratch indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:finish];bind({d,alpha,per,scratch,status});[enc setBytes:&last length:4 atIndex:5];[enc setBytes:&budget length:4 atIndex:6];[enc setBuffer:mesh offset:0 atIndex:7];one();[enc endEncoding];[cb commit];[cb waitUntilCompleted];return cb.status==MTLCommandBufferStatusCompleted;
    };
    auto* a=(float*)alpha.contents;auto* args=(uint32_t*)scratch.contents;auto* st=(NMMatterStatusGPU*)status.contents;
    *a=.625f;
    // J(0)=2.25, J(1)=2.43, J(.625)=2.53125. Both endpoints pass.
    if(!dispatchBlock(0,0,16)||*a!=.3125f||args[4]!=1||st->code)return 5;
    if(!dispatchBlock(1,0,16)||*a!=.3125f||args[4]!=0||st->code)return 6;
    if(!dispatchBlock(2,1,16)||args[0]!=0||st->code)return 7;
    *a=.625f;
    if(!dispatchBlock(0,1,16)||!st->code||*a!=.625f)return 8;
    std::printf("device=%s capped_trial=.625 recovered_trial=.3125 all_validated=1 completed_groups=0 final_invalid_rejected=1\n",gpu.name.UTF8String);
    return 0;
}}
