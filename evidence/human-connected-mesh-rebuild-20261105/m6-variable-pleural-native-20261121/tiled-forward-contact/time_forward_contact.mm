#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
namespace fs=std::filesystem;
int main(int argc,char** argv){@autoreleasepool{
    if(argc!=4)return 2;
    const fs::path a=argv[2],input=argv[3];
    auto dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
    auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    auto pipeline=[&](NSString* name){auto p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];if(!p)throw std::runtime_error("pipeline");return p;};
    auto load=[&](fs::path p){auto d=[NSData dataWithContentsOfFile:@(p.c_str())];if(!d.length)throw std::runtime_error(p.string());return [dev newBufferWithBytes:d.bytes length:d.length options:MTLResourceStorageModeShared];};
    auto alloc=[&](size_t n){auto b=[dev newBufferWithLength:n options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("alloc");std::memset(b.contents,0,n);return b;};
    auto tri=load(a/"cortical-triangles.bin"),faces=load(a/"forward-cortical-faces.bin"),prepared=alloc(tri.length);
    auto surface=load(input/"material_vertex.bin"),cortex=load(input/"cortical_vertex.bin"),frames=load(input/"frames.bin");
    auto pool=load(a/"pool-material-points.bin"),neighbors=load(a/"pool-cortical-neighborhood.bin"),sites=load(a/"forward-surface-samples.bin");
    auto dummy=alloc(64),needed=alloc(8);((uint32_t*)needed.contents)[0]=((uint32_t*)needed.contents)[1]=1;
    const uint32_t count=uint32_t(sites.length/32),triangles=uint32_t(tri.length/160),poolSize=uint32_t(pool.length/112);
    auto serial=pipeline(@"nm_pelvic_forward_query"),near=pipeline(@"nm_pelvic_forward_neighborhood"),bank=pipeline(@"nm_pelvic_forward_bank");
    auto clear=pipeline(@"nm_pelvic_clear_forward_dispatch"),transform=pipeline(@"nm_pelvic_forward_triangles");
    auto queue=[dev newCommandQueue];auto args=alloc(12),ids=alloc(count*8);
    auto g0=alloc(count*4),g1=alloc(count*4),x0=alloc(count*16),x1=alloc(count*16);
    {
        auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:transform];[e setBytes:&triangles length:4 atIndex:0];
        std::array<id<MTLBuffer>,5> b{faces,cortex,tri,frames,prepared};
        for(size_t k=0;k<b.size();++k)[e setBuffer:b[k] offset:0 atIndex:k+1];
        [e dispatchThreads:MTLSizeMake(triangles,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 3;
    }
    std::cout<<"pair,method,gpu_ms_per_query,sites,full_bank_sites\n";
    for(unsigned pair=0;pair<5;++pair)for(unsigned pass=0;pass<2;++pass){
        bool tiled=(pass^(pair&1))!=0;auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        auto gap=tiled?g1:g0;auto result=tiled?x1:x0;constexpr unsigned repeats=12;
        uint32_t d[4]={0,poolSize,0,0};
        for(unsigned j=0;j<repeats;++j){
            if(tiled){[e setComputePipelineState:clear];[e setBuffer:args offset:0 atIndex:0];[e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[e memoryBarrierWithScope:MTLBarrierScopeBuffers];}
            [e setComputePipelineState:tiled?near:serial];[e setBytes:d length:16 atIndex:0];
            std::array<id<MTLBuffer>,11> b{pool,surface,dummy,dummy,prepared,neighbors,needed,gap,sites,frames,result};
            for(size_t k=0;k<b.size();++k)[e setBuffer:b[k] offset:0 atIndex:k+1];
            if(tiled){[e setBuffer:ids offset:0 atIndex:12];[e setBuffer:args offset:0 atIndex:13];}
            [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if(tiled){[e setComputePipelineState:bank];std::array<id<MTLBuffer>,7> c{ids,pool,surface,sites,prepared,gap,result};
                for(size_t k=0;k<c.size();++k)[e setBuffer:c[k] offset:0 atIndex:k];
                [e dispatchThreadgroupsWithIndirectBuffer:args indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e memoryBarrierWithScope:MTLBarrierScopeBuffers];}
        }
        [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 4;
        if(pass && (memcmp(g0.contents,g1.contents,g0.length)||memcmp(x0.contents,x1.contents,x0.length)))throw std::runtime_error("forward distances or normals differ");
        std::cout<<pair<<','<<(tiled?"tiled":"serial")<<','<<1000*(cb.GPUEndTime-cb.GPUStartTime)/repeats<<','<<count<<','<<(tiled?*(uint32_t*)args.contents:0)<<'\n';
    }
    return 0;
}}
