#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int argc,const char** argv){
    @autoreleasepool {
        if(argc!=5)return 2;
        id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        NSError* error=nil;
        id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        id<MTLFunction> fn=[lib newFunctionWithName:@"registered_volume_test"];
        id<MTLComputePipelineState> p=fn?[device newComputePipelineStateWithFunction:fn error:&error]:nil;
        if(!p){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 1;}
        NSData* nodes=[NSData dataWithContentsOfFile:@(argv[2])];
        NSData* bounds=[NSData dataWithContentsOfFile:@(argv[3])];
        NSUInteger count=bounds.length/16;
        if(!count || bounds.length%16 || nodes.length!=count*4*96)return 2;
        id<MTLBuffer> v=[device newBufferWithBytes:nodes.bytes length:nodes.length options:MTLResourceStorageModeShared];
        id<MTLBuffer> b=[device newBufferWithBytes:bounds.bytes length:bounds.length options:MTLResourceStorageModeShared];
        id<MTLBuffer> o=[device newBufferWithLength:count*16 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue=[device newCommandQueue];
        id<MTLCommandBuffer> cb=[queue commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:p];[e setBuffer:v offset:0 atIndex:0];[e setBuffer:b offset:0 atIndex:1];[e setBuffer:o offset:0 atIndex:2];
        [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(MIN(p.threadExecutionWidth,count),1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);return 1;}
        return [[NSData dataWithBytes:o.contents length:count*16] writeToFile:@(argv[4]) atomically:YES]?0:1;
    }
}
