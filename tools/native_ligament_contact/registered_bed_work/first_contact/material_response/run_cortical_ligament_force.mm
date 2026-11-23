#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include "numi/matter/numi_human_shared.h"
int main(int argc,char**argv) {
    @autoreleasepool {
        if(argc!=4)return 2;
        NSString* folder=[NSString stringWithUTF8String:argv[2]];
        NSError* error=nil;id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        id<MTLFunction> fn=[library newFunctionWithName:@"numi_matter_metal::nm_numi_human_cortical_ligament_force"];
        if(!fn){fprintf(stderr,"Missing compiled force kernel\n");return 3;}
        id<MTLComputePipelineState> pipe=[device newComputePipelineStateWithFunction:fn error:&error];
        if(!pipe){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 4;}
        id<MTLBuffer> buffers[6];NSUInteger count=0;
        NSData* dims=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:@"dimensions.bin"]];
        if(dims.length!=2*sizeof(unsigned))return 5;
        const unsigned* shape=(const unsigned*)dims.bytes;
        NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
        dispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        dispatch.environmentCount=shape[0];dispatch.bodyPoseStride=shape[1];
        if(!shape[0]||!shape[1])return 5;
        NSArray* files=@[@"fibers.bin",@"ranges.bin",@"edges.bin",@"poses.bin"];
        for(unsigned i=0;i<4;++i){
            NSData* data=[NSData dataWithContentsOfFile:[folder stringByAppendingPathComponent:files[i]]];
            if(!data || !data.length)return 5;
            if(i==0){if(data.length%80)return 5;count=data.length/80;dispatch.passiveLigamentCount=uint32_t(count);}
            if((i==1 && data.length!=16*count)||(i==2 && data.length%64)||(i==3 && data.length!=32ull*shape[0]*shape[1]))return 5;
            buffers[i]=[device newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
        }
        count*=dispatch.environmentCount;
        buffers[4]=[device newBufferWithLength:48*count options:MTLResourceStorageModeShared];
        buffers[5]=[device newBufferWithLength:4*count options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
        id<MTLComputeCommandEncoder> enc=[command computeCommandEncoder];
        [enc setComputePipelineState:pipe];for(unsigned i=0;i<6;++i)[enc setBuffer:buffers[i] offset:0 atIndex:i];
        [enc setBytes:&dispatch length:sizeof(dispatch) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(dispatch.passiveLigamentCount,dispatch.environmentCount,1) threadsPerThreadgroup:MTLSizeMake(dispatch.passiveLigamentCount,MIN(dispatch.environmentCount,8u),1)];
        [enc endEncoding];[command commit];[command waitUntilCompleted];
        if(command.error){fprintf(stderr,"%s\n",command.error.localizedDescription.UTF8String);return 6;}
        NSData* result=[NSData dataWithBytes:buffers[4].contents length:48*count];
        if(![result writeToFile:[NSString stringWithUTF8String:argv[3]] atomically:YES])return 7;
        NSData* features=[NSData dataWithBytes:buffers[5].contents length:4*count];
        NSString* featuresPath=[[NSString stringWithUTF8String:argv[3]] stringByAppendingString:@".features.bin"];
        if(![features writeToFile:featuresPath atomically:YES])return 8;
        printf("%s %lu force evaluations %.9f GPU seconds\n",device.name.UTF8String,(unsigned long)count,command.GPUEndTime-command.GPUStartTime);
    }
}
