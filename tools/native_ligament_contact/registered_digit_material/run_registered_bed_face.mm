#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
int main(int argc, const char** argv) {
    @autoreleasepool {
        if (argc!=6) return 2;
        auto path=[&](int n){return [NSString stringWithUTF8String:argv[n]];};
        NSError* error=nil;
        auto gpu=MTLCreateSystemDefaultDevice();
        auto lib=[gpu newLibraryWithURL:[NSURL fileURLWithPath:path(1)] error:&error];
        auto fn=[lib newFunctionWithName:@"registered_bed_face"];
        auto pipe=fn?[gpu newComputePipelineStateWithFunction:fn error:&error]:nil;
        auto data=[NSData dataWithContentsOfFile:path(2) options:NSDataReadingMappedIfSafe error:&error];
        float args[]={strtof(argv[4],nullptr),strtof(argv[5],nullptr)};
        if (!pipe || pipe.threadExecutionWidth!=32 || data.length==0 || data.length%32 ||
            !std::isfinite(args[0]) || args[0]<=0 || !std::isfinite(args[1]) || args[1]<=0) {
            std::fprintf(stderr,"Invalid pipeline, face array or arguments (%s)\n", error?error.localizedDescription.UTF8String:"");
            return 3;
        }
        auto count=data.length/32;
        auto src=[gpu newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
        auto result=[gpu newBufferWithLength:count*48 options:MTLResourceStorageModeShared];
        auto queue=[gpu newCommandQueue];auto cmd=[queue commandBuffer];auto enc=[cmd computeCommandEncoder];
        if (!src || !result || !enc) return 4;
        [enc setComputePipelineState:pipe];
        [enc setBuffer:src offset:0 atIndex:0];[enc setBuffer:result offset:0 atIndex:1];[enc setBytes:args length:sizeof(args) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[cmd commit];[cmd waitUntilCompleted];
        if (cmd.status!=MTLCommandBufferStatusCompleted) return 5;
        auto out=[NSData dataWithBytesNoCopy:result.contents length:result.length freeWhenDone:NO];
        if (![out writeToFile:path(3) options:NSDataWritingAtomic error:&error]) return 6;
        std::printf("%s %zu faces %.6f ms\n", gpu.name.UTF8String,(size_t)count,1e3*(cmd.GPUEndTime-cmd.GPUStartTime));
    }
}
