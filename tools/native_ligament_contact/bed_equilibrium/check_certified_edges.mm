#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
int main(int argc,const char** argv){@autoreleasepool {
    if(argc!=3)return 2;
    auto file=[](const char* name){return [NSURL fileURLWithPath:[NSString stringWithUTF8String:name]];};
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError* error=nil;
    NSData* data=[NSData dataWithContentsOfURL:file(argv[2])];
    auto lib=[dev newLibraryWithURL:file(argv[1]) error:&error];
    auto function=[lib newFunctionWithName:@"checkCertifiedEdges"];
    if(!data||!data.length||data.length%(5*sizeof(float))||!function)return 3;
    auto pipe=[dev newComputePipelineStateWithFunction:function error:&error];
    if(!pipe)return 4;
    const auto count=data.length/(5*sizeof(float));
    auto in=[dev newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
    auto out=[dev newBufferWithLength:count*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if(!in||!out)return 5;
    memset(out.contents,0xff,out.length);
    auto queue=[dev newCommandQueue];auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];[enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(pipe.threadExecutionWidth,1,1)];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)return 6;
    auto values=static_cast<const uint32_t*>(out.contents);size_t failures=0;
    for(size_t i=0;i<count;++i)failures+=values[i]!=0;
    printf("{\"device\":\"%s\",\"proved_edges_executed\":%zu,\"mismatches\":%zu}\n",dev.name.UTF8String,size_t(count),failures);
    return failures?1:0;
}}
