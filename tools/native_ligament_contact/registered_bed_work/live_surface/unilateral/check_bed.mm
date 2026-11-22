#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int argc,const char** argv) {
    @autoreleasepool {
        if(argc!=4)return 2;
        auto path=[&](int n){return [NSString stringWithUTF8String:argv[n]];};
        auto gpu=MTLCreateSystemDefaultDevice();NSError* error=nil;
        auto lib=[gpu newLibraryWithURL:[NSURL fileURLWithPath:path(1)] error:&error];
        auto fn=[lib newFunctionWithName:@"measure_bed_projection"];
        auto pipe=[gpu newComputePipelineStateWithFunction:fn error:&error];
        auto data=[NSData dataWithContentsOfFile:path(2)];
        if(!pipe || !data.length || data.length%80)return 3;
        auto in=[gpu newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
        auto out=[gpu newBufferWithLength:(data.length/80)*64 options:MTLResourceStorageModeShared];
        auto q=[gpu newCommandQueue];auto cb=[q commandBuffer];auto enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];[enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(data.length/80,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted)return 4;
        auto result=[NSData dataWithBytes:out.contents length:out.length];
        if(![result writeToFile:path(3) options:NSDataWritingAtomic error:&error])return 5;
        std::puts(gpu.name.UTF8String);
    }
}
