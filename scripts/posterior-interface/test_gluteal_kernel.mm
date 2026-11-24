#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/numi_human_shared.h"
#include "metalrobo/engine_types.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc,char** argv) {
    @autoreleasepool {
        if (argc != 5) return 2;
        id<MTLDevice> d=MTLCreateSystemDefaultDevice();
        NSError* error=nil;
        id<MTLLibrary> lib=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        id<MTLFunction> f=[lib newFunctionWithName:@"numi_matter_metal::nm_human_gluteal_interface"];
        id<MTLComputePipelineState> p=f ? [d newComputePipelineStateWithFunction:f error:&error] : nil;
        NSData* K=[NSData dataWithContentsOfFile:@(argv[2])];
        FILE* fixture=std::fopen(argv[3],"rb");
        if (!fixture || !p || K.length!=171*171*sizeof(float)) return 3;
        std::array<unsigned,51> ids;
        std::array<std::array<float,3>,54> positions;
        std::array<std::array<float,4>,3> quats;
        if (std::fread(ids.data(),sizeof(ids),1,fixture)!=1 ||
            std::fread(positions.data(),sizeof(positions),1,fixture)!=1 ||
            std::fread(quats.data(),sizeof(quats),1,fixture)!=1 || std::fgetc(fixture)!=EOF) return 4;
        std::fclose(fixture);
        auto buffer=[&](NSUInteger n) {return [d newBufferWithLength:n options:MTLResourceStorageModeShared];};
        std::array<id<MTLBuffer>,7> b{};
        NMNumiHumanTendonFEMLoadDispatchGPU args{};
        args.femNodeCount=321219;args.bodyPoseStride=146;args.environmentCount=1;
        b[1]=buffer(args.femNodeCount*sizeof(NMFEMNodeStateGPU));b[2]=buffer(146*sizeof(MRArticulatedBodyPoseGPU));
        b[3]=buffer(args.femNodeCount*sizeof(nm_float4));b[4]=buffer(146*sizeof(NMNumiHumanBodyWrenchGPU));
        b[5]=[d newBufferWithBytes:K.bytes length:K.length options:MTLResourceStorageModeShared];b[6]=buffer(512);
        auto* nodes=static_cast<NMFEMNodeStateGPU*>(b[1].contents);
        auto* poses=static_cast<MRArticulatedBodyPoseGPU*>(b[2].contents);
        const unsigned bodies[]={128,131,145};
        for(int i=0;i<51;++i) {
            if(ids[i]>=args.femNodeCount)return 5;
            std::memcpy(&nodes[ids[i]].positionAndMass,positions[i].data(),3*sizeof(float));
        }
        for(int i=0;i<3;++i) {
            std::memcpy(&poses[bodies[i]].position,positions[51+i].data(),3*sizeof(float));
            std::memcpy(&poses[bodies[i]].orientation,quats[i].data(),4*sizeof(float));
        }
        id<MTLCommandQueue> q=[d newCommandQueue];
        id<MTLCommandBuffer> cmd=[q commandBuffer];
        id<MTLComputeCommandEncoder> e=[cmd computeCommandEncoder];
        [e setComputePipelineState:p];[e setBytes:&args length:sizeof(args) atIndex:0];
        for(unsigned i=1;i<b.size();++i) [e setBuffer:b[i] offset:0 atIndex:i];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];[cmd commit];[cmd waitUntilCompleted];
        if(cmd.status!=MTLCommandBufferStatusCompleted)return 6;
        FILE* out=std::fopen(argv[4],"wb");if(!out)return 7;
        const auto* skin=static_cast<const nm_float4*>(b[3].contents);
        const auto* bone=static_cast<const NMNumiHumanBodyWrenchGPU*>(b[4].contents);
        for(auto id:ids) std::fwrite(&skin[id],sizeof(nm_float4),1,out);
        for(auto id:bodies) std::fwrite(&bone[id],sizeof(NMNumiHumanBodyWrenchGPU),1,out);
        std::fwrite(b[6].contents,128,1,out);
        return std::fclose(out)!=0;
    }
}
