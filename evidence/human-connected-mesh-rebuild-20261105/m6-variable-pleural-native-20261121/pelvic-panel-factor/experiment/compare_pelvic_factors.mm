#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace fs=std::filesystem;
int main(int argc,char** argv){
    @autoreleasepool {
        if(argc!=7)return 2;
        auto dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
        auto kernel=[&](const char* lib,NSString* fn){
            auto l=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(lib)] error:&err];
            auto k=[dev newComputePipelineStateWithFunction:[l newFunctionWithName:fn] error:&err];
            if(!k)throw std::runtime_error("factor pipeline");return k;};
        auto old=kernel(argv[1],@"nm_pelvic_bulk_factor"),panel=kernel(argv[2],@"nm_pelvic_panel_factor");
        fs::path assets=argv[3],parts=argv[4],inputs=argv[5],out=argv[6];
        auto blob=[&](fs::path p){NSData* d=[NSData dataWithContentsOfFile:@(p.c_str())];
            if(!d.length)throw std::runtime_error(p.string());return [dev newBufferWithBytes:d.bytes length:d.length options:MTLResourceStorageModeShared];};
        auto alloc=[&](size_t size){auto b=[dev newBufferWithLength:size options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("allocation");std::memset(b.contents,0,size);return b;};
        std::array<id<MTLBuffer>,13> a,b;
        const char* factors[]={"pelvic-factor-rows","pelvic-factor-columns","pelvic-factor-values","pelvic-factor-levels","pelvic-factor-order","pelvic-factor-perm","pelvic-factor-scale"};
        for(unsigned i=0;i<7;++i)a[i]=blob(assets/(std::string(factors[i])+".bin"));
        auto trial=blob(inputs);a[7]=trial;a[8]=alloc(trial.length);a[9]=alloc(8);
        ((unsigned*)a[9].contents)[0]=((unsigned*)a[9].contents)[1]=1;
        for(unsigned i=0;i<3;++i)a[10+i]=blob(assets/(std::string("pelvic-projector-")+std::array<const char*,3>{"rows","cols","vals"}[i]+".bin"));
        b=a;b[8]=alloc(trial.length);
        for(unsigned i=0;i<4;++i)b[i]=blob(parts/(std::string("factor-panel-")+std::array<const char*,4>{"spans","indices","values","coefficients"}[i]+".bin"));
        b[4]=blob(parts/"offsets.bin");
        auto queue=[dev newCommandQueue];
        auto run=[&](bool usePanel,unsigned n){
            auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
            [e setComputePipelineState:(usePanel?panel:old)];
            auto& table=usePanel?b:a;
            for(unsigned j=0;j<table.size();++j)[e setBuffer:table[j] offset:0 atIndex:j];
            for(unsigned i=0;i<n;++i){[e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e memoryBarrierWithScope:MTLBarrierScopeBuffers];}
            [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Metal factor failed");
            return 1000*(cb.GPUEndTime-cb.GPUStartTime)/n;};
        run(false,3);run(true,3);
        std::cout<<"pair,original_gpu_ms,panel_gpu_ms\n"<<std::setprecision(12);
        for(unsigned j=0;j<5;++j){
            double x,y;if(j%2){y=run(true,16);x=run(false,16);}else{x=run(false,16);y=run(true,16);}
            std::cout<<j<<','<<x<<','<<y<<'\n';}
        for(unsigned i=0;i<2;++i){auto& buffers=i?b:a;std::ofstream f(out/(i?"panel-output.bin":"original-output.bin"),std::ios::binary);
            f.write((const char*)buffers[8].contents,buffers[8].length);if(!f)throw std::runtime_error("factor output");}
    }
}
