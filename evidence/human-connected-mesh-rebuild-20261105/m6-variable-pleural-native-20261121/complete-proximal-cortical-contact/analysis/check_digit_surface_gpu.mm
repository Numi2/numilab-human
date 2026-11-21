#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
int main(int argc,char** argv){@autoreleasepool {
    if(argc!=4)return 2;
    std::ifstream input(argv[2],std::ios::binary|std::ios::ate);
    const auto bytes=input.tellg();if(bytes<=0 || size_t(bytes)%96)return 3;
    const auto counts=size_t(bytes)/96;std::vector<unsigned char> payload(bytes);
    input.seekg(0);if(!input.read((char*)payload.data(),bytes))return 4;
    auto device=MTLCreateSystemDefaultDevice();NSError* error=nil;
    auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    auto fn=[lib newFunctionWithName:@"nm_digit_surface_coupon"];
    auto pipe=fn?[device newComputePipelineStateWithFunction:fn error:&error]:nil;
    if(!pipe){std::cerr<<[[error localizedDescription] UTF8String];return 5;}
    auto src=[device newBufferWithBytes:payload.data() length:payload.size() options:MTLResourceStorageModeShared];
    auto out=[device newBufferWithLength:counts*48 options:MTLResourceStorageModeShared];
    auto queue=[device newCommandQueue];auto cb=[queue commandBuffer];
    auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipe];
    [e setBuffer:src offset:0 atIndex:0];[e setBuffer:out offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake(counts,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)return 6;
    std::ofstream file(argv[3],std::ios::binary);file.write((const char*)out.contents,out.length);
    return file.good()?0:7;
}}
