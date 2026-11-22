#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv){
    @autoreleasepool {
        if(argc!=2)return 2;
        auto dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
        auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
        auto fn=[lib newFunctionWithName:@"nm_pelvic_fit_tractions"];
        auto ps=[dev newComputePipelineStateWithFunction:fn error:&err];
        if(!ps)throw std::runtime_error("unavailable pressure-fit kernel");
        auto queue=[dev newCommandQueue];
        auto buffer=[&](size_t bytes){
            auto b=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if(!b)throw std::runtime_error("test allocation");
            std::memset(b.contents,0,bytes);return b;
        };
        std::array<id<MTLBuffer>,10> b={buffer(8),buffer(512*4),buffer(2*112),buffer(512*16),buffer(512*64),
            buffer(2380*16),buffer(2380*16),buffer(512*4),buffer(2380*8*4),buffer(2380*16)};
        auto ui=[&](int k){return static_cast<unsigned*>(b[k].contents);};
        auto fp=[&](int k){return static_cast<float*>(b[k].contents);};
        ui(0)[0]=1;
        std::fill(ui(1),ui(1)+512,~0u);ui(1)[0]=0;ui(1)[1]=1;
        // Unit contact interpolation at node zero. Both rows share a normal,
        // one is separated by 4.9 micrometres and the other touches the cortex.
        for(int i=0;i<2;++i){
            fp(2)[28*i+24]=1;fp(3)[4*i]=1;
        }
        ui(5)[2]=1;fp(6)[0]=1;ui(8)[0]=1;
        fp(4)[15]=4.9e-6f;fp(4)[16+15]=1.e-9f;
        fp(7)[0]=1;fp(7)[1]=0;fp(7)[256]=.25f;
        auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:ps];
        for(int i=0;i<10;++i)[enc setBuffer:b[i] offset:0 atIndex:i];
        unsigned mode=1;[enc setBytes:&mode length:sizeof(mode) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted || fp(7)[0]!=0 ||
           std::abs(fp(7)[1]-1.f)>1e-6f || fp(7)[256]!=.25f || std::abs(fp(9)[0])>1e-6f)
            throw std::runtime_error("unilateral recovery failed force, clearance or side isolation");
        std::cout<<"passed separated-row release, P1 force transfer to a touching row, free residual and inactive side isolation\n";
    }
}
