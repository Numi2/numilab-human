#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/numi_human_shared.h"
#include "metalrobo/engine_types.h"
#include <cstdio>
int main(int argc,char** argv) {
 @autoreleasepool {
    if(argc!=3)return 2;
    NSString* root=[NSString stringWithUTF8String:argv[2]];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); NSError* error=nil;
    id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    id<MTLCommandQueue> queue=[device newCommandQueue];
    auto input=[&](NSString* name) -> id<MTLBuffer> {
        NSData* d=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:name]];
        if(!d.length)return nil;
        return [device newBufferWithBytes:d.bytes length:d.length options:MTLResourceStorageModeShared];
    };
    auto serialize=[&](id<MTLBuffer> b, NSString* name){
        return [[NSData dataWithBytes:b.contents length:b.length] writeToFile:[root stringByAppendingPathComponent:name] atomically:YES];
    };
    id<MTLBuffer> fibers=input(@"fibers.bin"),poses=input(@"poses.bin"),jac=input(@"jacobians.bin");
    if(!fibers||!poses||!jac||fibers.length!=80||poses.length%(157*32))return 3;
    const unsigned n=poses.length/(157*32),dofs=12;
    if(jac.length!=4ull*157*12*dofs*n)return 3;
    NMNumiHumanTendonFEMLoadDispatchGPU d{};d.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
    d.passiveLigamentCount=1;d.environmentCount=n;d.bodyPoseStride=157;d.dofCount=dofs;
    d.pointJacobianStride=157*12*dofs;d.generalizedForceStride=dofs;
    auto buffer=[&](NSUInteger size){auto b=[device newBufferWithLength:size options:MTLResourceStorageModeShared];memset(b.contents,0,size);return b;};
    id<MTLBuffer> force=buffer(4ull*dofs*n),status=buffer(n*sizeof(MRMetalWorldStatusGPU)),audit=buffer(n*sizeof(NMNumiHumanPassiveLigamentAuditGPU)),zero=buffer(256);
    for(unsigned e=0;e<n;++e){auto& s=((MRMetalWorldStatusGPU*)status.contents)[e];s.environment=e;s.code=MR_STEP_SUCCESS;s.successfulSubsteps=1;}
    for(int phase=0;phase<2;++phase){
        NSString* name=phase?@"nm_numi_human_audit_passive_ligaments":@"nm_numi_human_apply_pre_source_rigid_corrections";
        auto func=[lib newFunctionWithName:[@"numi_matter_metal::" stringByAppendingString:name]];
        auto pipe=func?[device newComputePipelineStateWithFunction:func error:&error]:nil;
        if(!pipe){fprintf(stderr,"pipeline %s: %s\n",name.UTF8String,error.localizedDescription.UTF8String);return 4;}
        auto command=[queue commandBuffer];auto enc=[command computeCommandEncoder];[enc setComputePipelineState:pipe];
        [enc setBytes:&d length:sizeof(d) atIndex:0];
        NSArray* bindings=phase?@[fibers,poses,audit]:@[zero,zero,zero,poses,jac,force,zero,fibers,zero,zero,status];
        for(NSUInteger i=0;i<bindings.count;++i)[enc setBuffer:bindings[i] offset:0 atIndex:i+1];
        [enc dispatchThreads:MTLSizeMake(n*(phase?1:dofs),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[command commit];[command waitUntilCompleted];if(command.error)return 5;
    }
    return serialize(force,@"generalized.f32")&&serialize(audit,@"audit.f32")?0:6;
 }
}
