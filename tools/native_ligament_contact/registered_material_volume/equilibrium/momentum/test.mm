#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
int main(int argc,char** argv){@autoreleasepool {
    if(argc!=2)return 2;
    auto gpu=MTLCreateSystemDefaultDevice();NSError* error=nil;
    auto lib=[gpu newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    if(!lib)return 3;
    auto pipeline=[gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"testBalance"] error:&error];
    if(!pipeline)return 4;
    auto out=[gpu newBufferWithLength:80 options:MTLResourceStorageModeShared];
    auto queue=[gpu newCommandQueue];auto cmd=[queue commandBuffer]; auto enc=[cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];[enc setBuffer:out offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(5,1,1) threadsPerThreadgroup:MTLSizeMake(5,1,1)];[enc endEncoding];
    [cmd commit];[cmd waitUntilCompleted];if(cmd.status!=MTLCommandBufferStatusCompleted)return 5;
    auto rows=(float*)out.contents;
    int expected[]={9,6,8,8,9};bool passed=true;
    for(int i=0;i<5;i++){
        auto r=rows+i*4;std::printf("%d %.10g %.10g %.10g %.0f\n",i,r[0],r[1],r[2],r[3]);
        passed &= r[3]==expected[i];
        if(expected[i]==9)passed &= r[0]<.6f && r[1]>.6f && r[2]<1e-10f;
        if(expected[i]==6)passed &= r[0]==r[1] && r[2]<1e-10f;
    }
    return passed?0:1;
}}
