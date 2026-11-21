#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include "numi/matter/numi_human_shared.h"
#include <cstdio>
#include <vector>
#include <cstring>
#include <stdexcept>
int main(int argc,const char** argv) { @autoreleasepool {
    if(argc!=7) return 2;
    auto device=MTLCreateSystemDefaultDevice(); NSError* error=nil;
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    auto kernel=[library newFunctionWithName:@"numi_matter_metal::nm_numi_human_audit_passive_ligaments"];
    if(!kernel) return 3;
    auto pipeline=[device newComputePipelineStateWithFunction:kernel error:&error];
    auto poses=[NSData dataWithContentsOfFile:@(argv[3])];
    auto rows=[NSData dataWithContentsOfFile:@(argv[2])];
    if(!pipeline || !poses || rows.length!=sizeof(NMNumiHumanPassiveLigamentGPU)+sizeof(NMNumiHumanLigamentContactStripGPU)) return 4;
    NMNumiHumanTendonFEMLoadDispatchGPU d{};
    d.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
    d.bodyPoseStride=2;d.passiveLigamentCount=1;d.dofCount=12;d.generalizedForceStride=12;d.pointJacobianStride=8*3*12;
    d.environmentCount=static_cast<uint32_t>(poses.length/(2*sizeof(MRArticulatedBodyPoseGPU)));
    if(!d.environmentCount||poses.length!=2*d.environmentCount*sizeof(MRArticulatedBodyPoseGPU)) return 5;
    auto rb=[device newBufferWithBytes:rows.bytes length:rows.length options:MTLResourceStorageModeShared];
    auto pb=[device newBufferWithBytes:poses.bytes length:poses.length options:MTLResourceStorageModeShared];
    auto result=[device newBufferWithLength:d.environmentCount*sizeof(NMNumiHumanPassiveLigamentAuditGPU) options:MTLResourceStorageModeShared];
    auto q=[device newCommandQueue];auto command=[q commandBuffer];auto enc=[command computeCommandEncoder];
    [enc setComputePipelineState:pipeline];[enc setBytes:&d length:sizeof(d) atIndex:0];
    [enc setBuffer:rb offset:0 atIndex:1];[enc setBuffer:pb offset:0 atIndex:2];[enc setBuffer:result offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(d.environmentCount,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(d.environmentCount,32u),1,1)];
    [enc endEncoding];
    auto work=[library newFunctionWithName:@"numi_matter_metal::nm_numi_human_apply_pre_source_rigid_corrections"];
    auto wp=[device newComputePipelineStateWithFunction:work error:&error];
    auto jac=[NSData dataWithContentsOfFile:@(argv[5])];
    if(!wp || jac.length!=d.environmentCount*d.pointJacobianStride*sizeof(float)) return 8;
    auto jb=[device newBufferWithBytes:jac.bytes length:jac.length options:MTLResourceStorageModeShared];
    auto force=[device newBufferWithLength:d.environmentCount*12*sizeof(float) options:MTLResourceStorageModeShared];
    std::memset(force.contents,0,force.length);
    std::vector<MRMetalWorldStatusGPU> admit(d.environmentCount);
    for (uint32_t i=0;i<d.environmentCount;i++){admit[i].code=MR_STEP_SUCCESS;admit[i].environment=i;admit[i].successfulSubsteps=1;}
    admit.back().code=MR_STEP_DID_NOT_CONVERGE;
    auto vb=[device newBufferWithBytes:admit.data() length:admit.size()*sizeof(admit[0]) options:MTLResourceStorageModeShared];
    enc=[command computeCommandEncoder];[enc setComputePipelineState:wp];[enc setBytes:&d length:sizeof(d) atIndex:0];
    for(unsigned k=1;k<=11;++k)[enc setBuffer:rb offset:0 atIndex:k];
    [enc setBuffer:pb offset:0 atIndex:4];[enc setBuffer:jb offset:0 atIndex:5];[enc setBuffer:force offset:0 atIndex:6];
    [enc setBuffer:rb offset:0 atIndex:8];[enc setBuffer:vb offset:0 atIndex:11];
    [enc dispatchThreads:MTLSizeMake(d.environmentCount*12,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding];[command commit];[command waitUntilCompleted];
    if(command.status!=MTLCommandBufferStatusCompleted){NSLog(@"%@",command.error);return 6;}
    if(![[NSData dataWithBytes:result.contents length:result.length] writeToFile:@(argv[4]) atomically:YES]) return 7;
    if(![[NSData dataWithBytes:force.contents length:force.length] writeToFile:@(argv[6]) atomically:YES]) return 9;
    std::fprintf(stderr,"GPU %s, rows %u\n",device.name.UTF8String,d.environmentCount);
    return 0;
}}
