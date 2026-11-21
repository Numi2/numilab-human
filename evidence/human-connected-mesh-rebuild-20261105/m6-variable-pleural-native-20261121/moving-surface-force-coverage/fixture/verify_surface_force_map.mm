#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
static void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
struct alignas(16) ContactProbePoint { float prefix[20]; uint32_t nodes[4]; float shape[4]; };

int main(int argc,char** argv) { @autoreleasepool {
    if (argc!=4) return 2;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    NSError* err=nil;
    auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    auto kernel=[&](NSString* name) {
        auto p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
        require(p!=nil,"missing solver kernel");return p;
    };
    auto columns=kernel(@"nm_pelvic_schur_columns"), factor=kernel(@"nm_pelvic_bulk_factor");
    fs::path assets=argv[2];
    auto load=[&](fs::path p) {
        auto bytes=[NSData dataWithContentsOfFile:@(p.c_str())];
        require(bytes.length>0,"missing asset");
        auto b=[dev newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];
        require(b!=nil,"asset allocation");return b;
    };
    auto alloc=[&](size_t len) {
        auto b=[dev newBufferWithLength:len options:MTLResourceStorageModeShared];
        require(b!=nil,"buffer allocation");std::memset(b.contents,0,len);return b;
    };
    auto old=load(fs::path(argv[3])/"cortical-free-index.bin");
    auto map=load(assets/"cortical-free-index.bin"), cache=load(assets/"cortical-flexibility.bin");
    auto nodes=unsigned(map.length/4), half=nodes/2;
    auto spans=load(assets/"free-node-rows.bin"), proj=load(assets/"pelvic-projector-rows.bin");
    require(old.length==map.length && spans.length==16*nodes,"material layout mismatch");
    auto slots=alloc(512*4), pool=alloc(2*sizeof(ContactProbePoint)), normals=alloc(512*16), needed=alloc(8);
    auto response=alloc(size_t(nodes)*256*16), rhs=alloc(nodes*16), delta=alloc(nodes*16);
    std::memset(slots.contents,0xff,slots.length);
    std::fill_n((uint32_t*)needed.contents,2,1u);
    const char* names[]={"pelvic-factor-rows","pelvic-factor-columns","pelvic-factor-values","pelvic-factor-levels",
                         "pelvic-factor-order","pelvic-factor-perm","pelvic-factor-scale"};
    std::array<id<MTLBuffer>,13> b;
    for (int i=0;i<7;++i) b[i]=load(assets/(std::string(names[i])+".bin"));
    b[7]=rhs; b[8]=delta; b[9]=needed; b[10]=proj;
    b[11]=load(assets/"pelvic-projector-cols.bin");b[12]=load(assets/"pelvic-projector-vals.bin");
    auto q=[dev newCommandQueue];
    std::cout<<"probe,side,owner_node,relative_l2,max_abs_m_per_N,reference_l2_m_per_N\n";
    for (int mode=0;mode<3;++mode) {
        std::memset(rhs.contents,0,rhs.length);std::memset(delta.contents,0,delta.length);
        for (uint32_t side=0;side<2;++side) {
            uint32_t chosen=UINT32_MAX;
            for(uint32_t i=side*half;i<(side+1)*half;++i) {
                require(((uint32_t*)spans.contents)[4*i+3]==side,"unequal regional partition");
                bool free=((uint32_t*)spans.contents)[4*i+2]!=0;
                bool floating=((uint32_t*)proj.contents)[6*i+1]!=0;
                if(((uint32_t*)old.contents)[i]==UINT32_MAX && ((uint32_t*)map.contents)[i]!=UINT32_MAX &&
                   ((mode==0 && free && !floating) || (mode==1 && !free) || (mode==2 && floating))) {chosen=i;break;}
            }
            require(chosen!=UINT32_MAX,"missing source probe");
            ContactProbePoint& p=((ContactProbePoint*)pool.contents)[side];
            std::fill_n(p.nodes,4,chosen);p.shape[0]=1;
            ((uint32_t*)slots.contents)[256*side]=side;
            const float normal[3]={.36f,-.48f,.8f};
            std::memcpy((float*)normals.contents+4*256*side,normal,sizeof(normal));
            std::memcpy((float*)rhs.contents+4*chosen,normal,sizeof(normal));
        }
        auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];
        for (uint32_t side=0;side<2;++side) {
            [e setComputePipelineState:columns];uint32_t args[4]={side,1,0,0};
            [e setBytes:args length:16 atIndex:0];
            std::array<id<MTLBuffer>,7> in={slots,pool,normals,map,cache,response,needed};
            for(size_t i=0;i<in.size();++i)[e setBuffer:in[i] offset:0 atIndex:1+i];
            [e dispatchThreads:MTLSizeMake(half,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        }
        [e setComputePipelineState:factor];
        for(size_t i=0;i<b.size();++i)[e setBuffer:b[i] offset:0 atIndex:i];
        [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"GPU failure");
        for(uint32_t side=0;side<2;++side) {
            double diff=0,ref=0,maximum=0;
            for(unsigned i=0;i<half;++i)for(unsigned k=0;k<3;++k) {
                double x=((float*)response.contents)[4*(half*256*side+i)+k],y=((float*)delta.contents)[4*(half*side+i)+k];
                require(std::isfinite(x)&&std::isfinite(y),"nonfinite contact response");
                diff+=(x-y)*(x-y);ref+=y*y;maximum=std::max(maximum,std::abs(x-y));
            }
            double relative=ref ? std::sqrt(diff/ref) : 0;
            std::cout<<mode<<','<<side<<','<<((ContactProbePoint*)pool.contents)[side].nodes[0]<<','<<relative<<','<<maximum<<','<<std::sqrt(ref)<<'\n';
            require(ref ? relative<1.e-3 : maximum==0.,"GPU compliance and regional solve disagree");
        }
    }
    return 0;
} }
