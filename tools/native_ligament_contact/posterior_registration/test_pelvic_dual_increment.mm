#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

int main(int argc,char**argv){
    @autoreleasepool {
        if(argc!=3&&argc!=4)return 2;
        unsigned perSide=std::stoul(argv[2]),count=2*perSide;
        if(perSide<2||perSide>16000)return 2;
        auto device=MTLCreateSystemDefaultDevice();
        NSError* error=nil;
        auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if(!lib){NSLog(@"%@",error);return 2;}
        auto pipeline=[&](NSString* name){auto fn=[lib newFunctionWithName:name];
            if(!fn)throw std::runtime_error("missing incremental contact kernel");
            auto p=[device newComputePipelineStateWithFunction:fn error:&error];
            if(!p)throw std::runtime_error(error.localizedDescription.UTF8String);return p;};
        auto zero=[&](size_t bytes){auto b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            std::memset(b.contents,0,bytes);return b;};
        auto needed=zero(8),slots=zero(512*4),old=zero(512*4),trial=zero(512*4);
        auto gram=zero(2*65536*4),rows=zero(512*64),coordinates=zero(count*16);
        auto response=zero(size_t(count)*256*16),material=zero(8);
        auto uints=[](id<MTLBuffer> b){return static_cast<unsigned*>(b.contents);};
        auto floats=[](id<MTLBuffer> b){return static_cast<float*>(b.contents);};
        uints(needed)[0]=1;std::memset(slots.contents,0xff,slots.length);
        uints(slots)[0]=3;uints(slots)[1]=4;
        floats(old)[0]=2;floats(old)[1]=1;floats(trial)[0]=1.5f;floats(trial)[1]=2.25f;
        floats(trial)[300]=22;floats(gram)[0]=2;floats(gram)[1]=floats(gram)[256]=.5f;floats(gram)[257]=1;
        floats(rows)[3]=.6f;floats(rows)[19]=-.2f;
        floats(response)[0]=1;floats(response)[1]=-.5f;
        floats(response)[4*perSide+1]=.25f;floats(response)[4*perSide+2]=2;
        floats(coordinates)[0]=2;floats(coordinates)[1]=-1;floats(coordinates)[2]=3;
        auto queue=[device newCommandQueue];auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        auto bind=[&](unsigned i,id<MTLBuffer> b){[e setBuffer:b offset:0 atIndex:i];};
        auto run=[&](unsigned n){[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e memoryBarrierWithScope:MTLBarrierScopeBuffers];};
        [e setComputePipelineState:pipeline(@"nm_pelvic_shift_dual_gap")];
        bind(0,needed);bind(1,slots);bind(2,old);bind(3,gram);bind(4,rows);run(512);
        [e setComputePipelineState:pipeline(@"nm_pelvic_dual_move")];
        [e setBytes:&count length:4 atIndex:0];bind(1,needed);bind(2,old);bind(3,trial);bind(4,response);bind(5,coordinates);
        if(argc==4)bind(6,material);run(count);
        [e setComputePipelineState:pipeline(@"nm_pelvic_dual_load")];bind(0,needed);bind(1,old);bind(2,trial);
        if(argc==4)bind(3,material);run(512);
        [e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted){NSLog(@"%@",cb.error);return 2;}
        const std::array<float,8> got{floats(rows)[3],floats(rows)[19],floats(coordinates)[0],floats(coordinates)[1],floats(coordinates)[2],floats(trial)[0],floats(trial)[1],floats(trial)[300]};
        const std::array<float,8> expect{-3.9f,-2.2f,1.75f,-.71875f,4.25f,1.75f,1.625f,22};
        for(size_t i=0;i<got.size();++i)if(!std::isfinite(got[i])||std::abs(got[i]-expect[i])>1e-6f){
            std::cerr<<"component "<<i<<" measured "<<got[i]<<" expected "<<expect[i]<<'\n';return 1;}
        std::cout<<"gap,paired_displacement,paired_pressure,inactive_side\npass,pass,pass,pass\n";
    }
}
