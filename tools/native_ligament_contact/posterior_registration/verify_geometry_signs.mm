#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv){
    @autoreleasepool{
        if(argc!=4)return 2;
        std::ifstream f(argv[2],std::ios::binary|std::ios::ate);
        auto bytes=f.tellg();
        if(bytes<=0||bytes%96||bytes>64*1024*1024)return 2;
        std::vector<char> p(bytes);f.seekg(0);if(!f.read(p.data(),bytes))return 2;
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
        auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
        auto pipeline=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"testGeometrySigns"] error:&err];
        if(!pipeline){NSLog(@"%@",err);return 1;}
        unsigned long n=p.size()/96;
        auto in=[dev newBufferWithBytes:p.data() length:p.size() options:MTLResourceStorageModeShared];
        auto out=[dev newBufferWithLength:n*16 options:MTLResourceStorageModeShared];
        auto queue=[dev newCommandQueue];auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];[enc setBuffer:in offset:0 atIndex:0];[enc setBuffer:out offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted){NSLog(@"%@",cb.error);return 1;}
        std::ofstream w(argv[3],std::ios::binary);w.write(static_cast<const char*>(out.contents),n*16);
        return !w.good();
    }
}
