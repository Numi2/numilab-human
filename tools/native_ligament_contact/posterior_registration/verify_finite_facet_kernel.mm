#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc,char** argv){
    @autoreleasepool {
        if(argc!=3)return 2;
        auto dev=MTLCreateSystemDefaultDevice();NSError* error=nil;
        auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        auto program=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_pelvic_finite_surface"] error:&error];
        if(!program){NSLog(@"%@",error);return 2;}
        std::ifstream input(argv[2],std::ios::binary);unsigned header[2]={};input.read((char*)header,8);
        unsigned n=header[0],first=header[1];if(n<1||n>64||first>1000000)return 2;
        auto read=[&](unsigned len){std::vector<unsigned char> bytes(len);input.read((char*)bytes.data(),len);
            if(!input)throw std::runtime_error("incomplete fixture");
            return [dev newBufferWithBytes:bytes.data() length:len options:MTLResourceStorageModeShared];};
        auto alloc=[&](unsigned len){return [dev newBufferWithLength:len options:MTLResourceStorageModeShared];};
        std::array<id<MTLBuffer>,9> b{read(n*17*16),read(n*10*16),read(n*4*16),read(146*32),read(146*16),read(8*16),
            alloc((first+n*2)*112),alloc((first+n*2)*4),alloc(8)};
        auto queue=[dev newCommandQueue];auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        [e setComputePipelineState:program];unsigned dims[]={0,n,0,0};[e setBytes:dims length:16 atIndex:0];
        for(unsigned i=0;i<b.size();++i)[e setBuffer:b[i] offset:0 atIndex:i+1];
        [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted){NSLog(@"%@",cb.error);return 2;}
        auto pool=static_cast<const float*>(b[6].contents),gap=static_cast<const float*>(b[7].contents);
        std::cout<<std::setprecision(9);
        for(unsigned i=first;i<first+2*n;++i){std::cout<<gap[i];for(unsigned k=0;k<4;++k)std::cout<<' '<<pool[28*i+24+k];std::cout<<'\n';}
    }
}
