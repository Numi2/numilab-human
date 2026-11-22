#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using Key=std::array<uint32_t,2>;
static void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}

int main(int argc,char** argv){
    @autoreleasepool {
        if(argc!=2)return 2;
        auto device=MTLCreateSystemDefaultDevice();NSError* error=nil;
        auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        require(lib!=nil,"library");auto queue=[device newCommandQueue];
        auto plan=[&](NSString* name){
            auto p=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];require(p!=nil,"kernel");return p;};
        auto reg=plan(@"nm_pelvic_motion_register"),phase=plan(@"nm_pelvic_motion_phase");
        auto buffer=[&](size_t size){auto b=[device newBufferWithLength:size options:MTLResourceStorageModeShared];require(b!=nil,"allocation");std::memset(b.contents,0,size);return b;};
        auto met=buffer(20),obs=buffer(512*sizeof(Key)),keys=buffer(512*sizeof(Key)),live=buffer(512*4),saved=buffer(512*sizeof(Key));
        auto m=(unsigned*)met.contents;auto hit=(Key*)obs.contents;auto work=(Key*)keys.contents;auto good=(Key*)saved.contents;auto load=(unsigned*)live.contents;
        auto dispatch=[&](id<MTLComputePipelineState> pipeline,std::vector<id<MTLBuffer>> inputs,bool phaseCall,unsigned step){
            auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipeline];
            unsigned idx=0;if(phaseCall){std::array<unsigned,4> d{0,step,0,0};[enc setBytes:d.data() length:16 atIndex:idx++];}
            for(auto b:inputs)[enc setBuffer:b offset:0 atIndex:idx++];
            [enc dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
            require(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");};
        auto run=[&](){dispatch(reg,{met,obs,keys,live},false,0);};
        std::array<Key,3> original{Key{20652,13567},Key{4960,1132},Key{20652,13567}};
        for(int reverse=0;reverse<2;++reverse){
            std::memset(keys.contents,0,keys.length);std::memset(live.contents,0,live.length);
            std::copy(original.begin(),original.end(),hit);if(reverse)std::reverse(hit,hit+3);
            hit[256]={30000,18000};m[0]=3;m[1]=1;m[2]=0;run();
            require(work[0]==Key{4961,1133} && work[1]==Key{20653,13568} && work[256]==Key{30001,18001},"deterministic bilateral attribution");
            require(!work[2][0] && load[0] && load[1] && load[256] && !m[2],"deduplication");
        }
        std::copy(work,work+512,good);std::memset(load,0,512*4);load[1]=1;
        dispatch(phase,{met,saved,keys,live},true,1);
        require(work[1]==good[1] && work[0]==Key{} && work[256]==Key{} && !m[0] && !m[1],"rollback restores only loaded rows");
        dispatch(phase,{met,saved,keys,live},true,0);
        require(work[1]==Key{},"fresh scene ignores past keys");
        for(unsigned j=0;j<256;++j){work[j]={j+1,22};hit[j]={512-j,21};}
        m[0]=256;run();require(m[2]==2,"working slots exhausted");
        std::cout<<"order,duplicate,bilateral,rollback,cold_start,capacity\npass,pass,pass,pass,pass,pass\n";
    }
}
