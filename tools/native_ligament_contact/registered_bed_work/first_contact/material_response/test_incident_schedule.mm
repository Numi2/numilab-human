#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
using Nodes=std::array<uint32_t,4>;
static bool share(const Nodes& a,const Nodes& b){
    for(auto v:a) if(std::find(b.begin(),b.end(),v)!=b.end()) return true;
    return false;
}
int main(int argc,const char** argv){ @autoreleasepool{
    if(argc!=2)return 2;
    NSError* error=nil; id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"checkColors"] error:&error];
    if(!pipeline){NSLog(@"%@",error);return 3;}
    auto queue=[device newCommandQueue];std::mt19937 rng(20261124);unsigned cases=0;
    for(unsigned live:{0u,1u,6u,28u,31u,32u,33u})for(unsigned pattern=0;pattern<4;pattern++){
        constexpr unsigned n=64;
        std::vector<Nodes> tet(n);std::vector<uint32_t> rows(n+33,0xffffffffu),baseline(n,0xffffffffu);
        for(unsigned i=0;i<n;i++) {
            tet[i]={4*i,4*i+1,4*i+2,4*i+3};rows[i]=i;
            if(pattern==1)tet[i][0]=10000;
            if(pattern==2)tet[i][0]=10000+i%5;
            if(pattern==3){tet[i][0]=10000+i%5;tet[i][1]=20000+(i+2)%7;}
        }
        if(live<=32)for(unsigned i=0;i<live;i++){
            uint32_t used=0;
            for(unsigned j=0;j<i;j++)if(share(tet[i],tet[j]))used|=1u<<baseline[j];
            baseline[i]=__builtin_ctz(~used);
        }
        rows[n]=live;
        for(unsigned i=0;i<std::min(live,32u);i++)rows[n+1+i]=i;
        for(unsigned trial=0;trial<4;trial++){
            std::shuffle(rows.begin()+n+1,rows.begin()+n+1+std::min(live,32u),rng);
            auto b=[device newBufferWithBytes:rows.data() length:rows.size()*4 options:MTLResourceStorageModeShared];
            auto t=[device newBufferWithBytes:tet.data() length:tet.size()*sizeof(Nodes) options:MTLResourceStorageModeShared];
            auto out=[device newBufferWithLength:n*4 options:MTLResourceStorageModeShared];
            std::fill_n((uint32_t*)out.contents,n,0xffffffffu);
            auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipeline];
            [enc setBuffer:b offset:0 atIndex:0];[enc setBuffer:t offset:0 atIndex:1];[enc setBuffer:out offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[enc endEncoding];
            [cb commit];[cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted || !std::equal(baseline.begin(),baseline.end(),(uint32_t*)out.contents))return 4;
            cases++;
        }
    }
    std::cout<<"device="<<device.name.UTF8String<<" matched_schedules="<<cases<<"\n";return 0;
}}
