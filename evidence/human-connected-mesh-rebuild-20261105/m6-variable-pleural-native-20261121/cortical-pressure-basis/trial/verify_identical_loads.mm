#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <array>
#include <cmath>

int main(int argc,char** argv){
    @autoreleasepool {
        if(argc!=2)return 2;
        auto dev=MTLCreateSystemDefaultDevice();NSError* error=nil;
        auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        auto plan=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_pelvic_merge_equal_loads"] error:&error];
        if(!plan)throw std::runtime_error("kernel");
        auto alloc=[&](size_t n){auto b=[dev newBufferWithLength:n options:MTLResourceStorageModeShared];std::memset(b.contents,0,n);return b;};
        std::array<id<MTLBuffer>,8> b{alloc(512*4),alloc(512*4),alloc(512*112),alloc(512*64),alloc(512*16),alloc(512*16),alloc(512*4),alloc(8)};
        auto slots=(unsigned*)b[0].contents;auto loads=(float*)b[1].contents;auto pools=(unsigned*)b[2].contents;auto row=(float*)b[3].contents;auto included=(unsigned*)b[6].contents;
        std::memset(slots,0xff,b[0].length);((unsigned*)b[7].contents)[0]=1;
        for(unsigned s:{0u,1u,2u,3u,4u,5u,6u,256u}){
            slots[s]=s;included[s]=1;loads[s]=static_cast<float>(s+1);
            auto site=pools+28*s;site[12]=128;site[13]=131;site[14]=(s==2?131:1);site[15]=1;
            bool perm=s==1;site[20]=perm?4:2;site[21]=perm?2:4;site[22]=100+s;site[23]=900+s;
            auto weights=(float*)(site+24);weights[0]=perm?.25f:.75f;weights[1]=perm?.75f:.25f;
            if(s==3)row[16*s+15]=std::nextafter(0.f,1.f)*1e30f;
            if(s==4)row[16*s+12]=.025f;
            if(s==5)weights[0]=std::nextafter(weights[0],1.f);
            if(s==6)((float*)b[4].contents)[4*s]=.5f;
        }
        auto q=[dev newCommandQueue];auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:plan];
        for(unsigned j=0;j<8;++j)[e setBuffer:b[j] offset:0 atIndex:j];
        [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted||loads[0]!=3||slots[0]!=0||slots[1]!=UINT32_MAX||loads[1]!=0||included[1]!=0)throw std::runtime_error("duplicate failed");
        for(unsigned s:{2u,3u,4u,5u,6u,256u})if(slots[s]!=s||loads[s]!=s+1||included[s]!=1)throw std::runtime_error("independent site changed");
        std::cout<<"parent_permutation,zero_weights,bone,gap,lever,weight,normal,inactive_side\npass,pass,pass,pass,pass,pass,pass,pass\n";
    }
}
