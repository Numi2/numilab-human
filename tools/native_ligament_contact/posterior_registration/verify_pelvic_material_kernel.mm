#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc,char** argv){
    @autoreleasepool {
        if(argc!=3)return 2;
        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
        NSError* error=nil;
        id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if(!lib){NSLog(@"%@",error);return 2;}
        auto fn=[lib newFunctionWithName:@"nm_pelvic_bulk_corners"];
        auto prog=[dev newComputePipelineStateWithFunction:fn error:&error];if(!prog)return 2;
        std::ifstream input(argv[2],std::ios::binary);
        unsigned n=0;input.read(reinterpret_cast<char*>(&n),4);if(n!=6)return 2;
        auto data=[&](size_t length){std::vector<unsigned char> v(length);input.read(reinterpret_cast<char*>(v.data()),length);
            if(!input)throw std::runtime_error("truncated material fixture");
            return [dev newBufferWithBytes:v.data() length:length options:MTLResourceStorageModeShared];};
        std::array<id<MTLBuffer>,6> b{data(16*n),data(48*n),data(64*n),data(4*n),
            [dev newBufferWithLength:64*n options:MTLResourceStorageModeShared],[dev newBufferWithLength:16*n options:MTLResourceStorageModeShared]};
        auto queue=[dev newCommandQueue];auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:prog];
        [e setBytes:&n length:4 atIndex:0];for(unsigned i=0;i<b.size();++i)[e setBuffer:b[i] offset:0 atIndex:i+1];
        [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted){NSLog(@"%@",cb.error);return 2;}
        auto force=static_cast<const float*>(b[4].contents);auto status=static_cast<const float*>(b[5].contents);
        std::cout<<std::setprecision(9);
        for(unsigned i=0;i<n;++i){
            for(unsigned j=0;j<4;++j)for(unsigned k=0;k<3;++k)std::cout<<force[16*i+4*j+k]<<' ';
            for(unsigned j=0;j<4;++j)std::cout<<status[4*i+j]<<' ';std::cout<<'\n';
        }
        return 0;
    }
}
