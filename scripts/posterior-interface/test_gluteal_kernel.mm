#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/numi_human_shared.h"
#include "metalrobo/engine_types.h"
#include <array>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc,char** argv) {
    @autoreleasepool {
        if (argc != 5 && argc != 6) return 2;
        id<MTLDevice> d=MTLCreateSystemDefaultDevice();
        NSError* error=nil;
        id<MTLLibrary> lib=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        id<MTLFunction> f=[lib newFunctionWithName:@"numi_matter_metal::nm_human_gluteal_interface"];
        id<MTLComputePipelineState> p=f ? [d newComputePipelineStateWithFunction:f error:&error] : nil;
        NSData* K=[NSData dataWithContentsOfFile:@(argv[2])];
        FILE* fixture=std::fopen(argv[3],"rb");
        if (!fixture || !p || K.length > 777u*777u*sizeof(float)) return 3;
        const unsigned dofs=static_cast<unsigned>(std::sqrt(double(K.length)/sizeof(float)));
        if (dofs<=18u || (dofs-18u)%3u || K.length!=dofs*dofs*sizeof(float)) return 3;
        const unsigned fatCount=(dofs-18u)/3u;
        std::vector<unsigned> ids(fatCount);
        std::vector<std::array<float,3>> positions(fatCount+3u);
        std::array<std::array<float,4>,3> quats;
        if (std::fread(ids.data(),sizeof(ids[0]),ids.size(),fixture)!=ids.size() ||
            std::fread(positions.data(),sizeof(positions[0]),positions.size(),fixture)!=positions.size() ||
            std::fread(quats.data(),sizeof(quats),1,fixture)!=1 || std::fgetc(fixture)!=EOF) return 4;
        std::fclose(fixture);
        auto buffer=[&](NSUInteger n) {return [d newBufferWithLength:n options:MTLResourceStorageModeShared];};
        std::array<id<MTLBuffer>,7> b{};
        NMNumiHumanTendonFEMLoadDispatchGPU args{};
        unsigned totalNodes=321219;
        if (argc==6) {
            char* end=nullptr;
            unsigned long value=std::strtoul(argv[5],&end,10);
            if (!end || end==argv[5] || *end || !value || value>10000000u) return 4;
            totalNodes=static_cast<unsigned>(value);
        }
        args.femNodeCount=totalNodes;args.bodyPoseStride=146;args.environmentCount=1;
        b[1]=buffer(args.femNodeCount*sizeof(NMFEMNodeStateGPU));b[2]=buffer(146*sizeof(MRArticulatedBodyPoseGPU));
        b[3]=buffer(args.femNodeCount*sizeof(nm_float4));b[4]=buffer(146*sizeof(NMNumiHumanBodyWrenchGPU));
        b[5]=[d newBufferWithBytes:K.bytes length:K.length options:MTLResourceStorageModeShared];b[6]=buffer(512);
        auto* nodes=static_cast<NMFEMNodeStateGPU*>(b[1].contents);
        auto* poses=static_cast<MRArticulatedBodyPoseGPU*>(b[2].contents);
        const unsigned bodies[]={128,131,145};
        for(unsigned i=0;i<fatCount;++i) {
            if(ids[i]>=args.femNodeCount)return 5;
            std::memcpy(&nodes[ids[i]].positionAndMass,positions[i].data(),3*sizeof(float));
        }
        for(int i=0;i<3;++i) {
            std::memcpy(&poses[bodies[i]].position,positions[fatCount+i].data(),3*sizeof(float));
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
