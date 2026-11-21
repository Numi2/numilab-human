#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv){@autoreleasepool{
    if(argc!=2)return 2;
    auto dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
    auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    auto pipe=[&](NSString* fn){auto k=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:fn] error:&err];if(!k)throw std::runtime_error("pipeline");return k;};
    auto bound=pipe(@"nm_pelvic_material_step_bound"),apply=pipe(@"nm_pelvic_material_step_apply");
    auto alloc=[&](size_t n){auto b=[dev newBufferWithLength:n options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("buffer");std::memset(b.contents,0,n);return b;};
    auto t=alloc(16),inv=alloc(48),u=alloc(64),proposal=alloc(64),needed=alloc(8),level=alloc(8),loads=alloc(2048),old=alloc(2048);
    for(unsigned k=0;k<4;++k)((uint32_t*)t.contents)[k]=k;
    for(unsigned k=0;k<3;++k)((float*)inv.contents)[5*k]=1.f;
    ((uint32_t*)needed.contents)[0]=1;
    struct Probe{float a[3];uint32_t k;};
    const Probe tests[]={{{1.25f,1,1},0},{{-1,-1,1},2},{{-1,1,1},2},{{8,1,1},1},{{NAN,1,1},24}};
    auto queue=[dev newCommandQueue];
    for(auto sample:tests){
        std::memset(u.contents,0,64);std::memset(proposal.contents,0,64);std::memset(level.contents,0,8);
        for(unsigned k=0;k<3;++k)((float*)proposal.contents)[4*(k+1)+k]=sample.a[k]-1.f;
        std::fill_n((float*)loads.contents,512,8.f);std::fill_n((float*)old.contents,512,4.f);
        auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        uint32_t n=1;[e setComputePipelineState:bound];[e setBytes:&n length:4 atIndex:0];
        std::array<id<MTLBuffer>,7>b{t,inv,u,u,proposal,needed,level};
        for(size_t j=0;j<b.size();++j)[e setBuffer:b[j] offset:0 atIndex:j+1];
        [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:apply];n=4;[e setBytes:&n length:4 atIndex:0];
        std::array<id<MTLBuffer>,6>c{u,proposal,loads,old,level,needed};
        for(size_t j=0;j<c.size();++j)[e setBuffer:c[j] offset:0 atIndex:j+1];
        [e dispatchThreads:MTLSizeMake(512,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];
        auto observed=*(uint32_t*)level.contents;
        double alpha=observed==24?0.:1./(1u<<observed);
        if(cb.status!=MTLCommandBufferStatusCompleted || observed!=sample.k ||
           ((float*)loads.contents)[0]!=4.+4.*alpha || ((float*)loads.contents)[256]!=8.f)return 1;
        for(unsigned k=0;k<3;++k) if(((float*)u.contents)[4*(k+1)+k]!=(alpha?alpha*(sample.a[k]-1.f):0.))return 1;
        std::cout<<observed<<','<<alpha<<','<<((float*)loads.contents)[0]<<'\n';
    }
    return 0;
}}
