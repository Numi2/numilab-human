#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Vec { float x=0,y=0,z=0,w=0; };
struct Site { uint32_t contact[20]{},nodes[4]{}; float weights[4]{}; };
struct Row { Vec normal,leverP,leverF,leverB; };
static_assert(sizeof(Site)==112 && sizeof(Row)==64);

int main(int argc,char** argv) {
    @autoreleasepool {
        if(argc!=3) return 2;
        const size_t nodes=std::stoul(argv[2]);
        if(nodes<4 || nodes>100000) return 2;
        auto dev=MTLCreateSystemDefaultDevice();
        NSError* err=nil;
        auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
        if(!lib){NSLog(@"%@",err);return 2;}
        auto fit=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_pelvic_fit_tractions"] error:&err];
        if(!fit){NSLog(@"%@",err);return 2;}
        auto queue=[dev newCommandQueue];
        for(int mode=0;mode<6;++mode) {
            std::array<unsigned,2> needed{1,0};
            std::array<unsigned,512> slot;
            slot.fill(UINT32_MAX);
            std::array<Site,2> sites{};
            std::array<Vec,512> normals{};
            std::array<Row,512> rows{};
            std::array<float,512> initial{};
            std::vector<unsigned> freeRow(4*nodes),mask(8*nodes);
            std::vector<Vec> internal(nodes),scratch(nodes);
            auto add=[&](unsigned i, unsigned a,unsigned b,float wa,float wb,Vec normal,float load,float gap) {
                slot[i]=i;sites[i].nodes[0]=a;sites[i].nodes[1]=b;
                sites[i].weights[0]=wa;sites[i].weights[1]=wb;normals[i]=normal;
                rows[i].leverB.w=gap;initial[i]=load;
                if(load>0){if(wa)mask[8*a]|=1u<<i;if(wb)mask[8*b]|=1u<<i;}
            };
            Vec nx{1,0,0,0},ny{0,1,0,0};
            freeRow[2]=freeRow[6]=1;
            add(0,0,1,.25f,.75f,nx,.25f,0);
            internal[0].x=.5f;internal[1].x=1.5f;
            float p0=2,p1=0;
            if(mode==1){add(1,0,1,.25f,.75f,ny,0,0);internal[0].y=.75f;internal[1].y=2.25f;p1=3;}
            if(mode==2){internal[0].x=-.5f;internal[1].x=-1.5f;p0=0;}
            if(mode==3){add(1,0,1,.25f,.75f,ny,0,.001f);internal[0].y=4;internal[1].y=2;}
            if(mode==4){freeRow[2]=0;internal[0].x=1.e5f;}
            if(mode==5){needed[0]=0;p0=.25f;}
            initial[300]=.75f;
            auto make=[&](const void* data,size_t bytes){
                return [dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
            };
            std::array<id<MTLBuffer>,10> buffers{
                make(needed.data(),sizeof(needed)),make(slot.data(),sizeof(slot)),make(sites.data(),sizeof(sites)),
                make(normals.data(),sizeof(normals)),make(rows.data(),sizeof(rows)),
                make(freeRow.data(),4*freeRow.size()),make(internal.data(),sizeof(Vec)*nodes),
                make(initial.data(),sizeof(initial)),make(mask.data(),4*mask.size()),make(scratch.data(),sizeof(Vec)*nodes)
            };
            auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:fit];
            for(unsigned b=0;b<buffers.size();++b)[e setBuffer:buffers[b] offset:0 atIndex:b];
            [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [e endEncoding];[cb commit];[cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted){NSLog(@"%@",cb.error);return 2;}
            const auto* actual=static_cast<const float*>(buffers[7].contents);
            if(!std::isfinite(actual[0])||!std::isfinite(actual[1])||std::abs(actual[0]-p0)>1e-5f||
               std::abs(actual[1]-p1)>1e-5f||actual[300]!=.75f) {
                std::cerr<<"fixture "<<mode<<" pressure "<<actual[0]<<" "<<actual[1]<<'\n';return 1;
            }
            std::cout<<mode<<','<<actual[0]<<','<<actual[1]<<','<<actual[300]<<'\n';
        }
    }
}
