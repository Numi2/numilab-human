#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc,char** argv){
    @autoreleasepool {
        if(argc!=3)return 2;
        auto dev=MTLCreateSystemDefaultDevice();NSError* error=nil;
        auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        bool actual=std::getenv("NUMI_VERIFY_ACTUAL_FACETS")!=nullptr;
        NSString* name=actual?@"nm_pelvic_finite_native":@"nm_pelvic_finite_surface";
        auto program=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];
        if(!program){NSLog(@"%@",error);return 2;}
        std::ifstream input(argv[2],std::ios::binary);unsigned header[2]={};input.read((char*)header,8);
        unsigned n=header[0],first=header[1];if(n<1||n>64||first>1000000)return 2;
        auto read=[&](unsigned len){std::vector<unsigned char> bytes(len);input.read((char*)bytes.data(),len);
            if(!input)throw std::runtime_error("incomplete fixture");
            return [dev newBufferWithBytes:bytes.data() length:len options:MTLResourceStorageModeShared];};
        auto alloc=[&](unsigned len){return [dev newBufferWithLength:len options:MTLResourceStorageModeShared];};
        std::array<id<MTLBuffer>,9> b{read(n*17*16),read(n*10*16),read(n*4*16),read(146*32),read(146*16),read(8*16),
            alloc((first+n*16)*112),alloc((first+n*16)*4),alloc(8)};
        auto queue=[dev newCommandQueue];auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        [e setComputePipelineState:program];unsigned dims[]={0,n,0,0};[e setBytes:dims length:16 atIndex:0];
        for(unsigned i=0;i<b.size();++i)[e setBuffer:b[i] offset:0 atIndex:i+1];
        if(actual){
            auto map=alloc(n*16),faces=alloc(n*16),bone=alloc(n*3*16),skin=alloc(n*3*16);
            auto mapping=static_cast<unsigned*>(map.contents),cortex=static_cast<unsigned*>(faces.contents);
            auto mv=static_cast<float*>(skin.contents),bv=static_cast<float*>(bone.contents);
            auto pairs=static_cast<const float*>(b[0].contents),tris=static_cast<const float*>(b[1].contents);
            auto delta=static_cast<const float*>(b[2].contents),poses=static_cast<const float*>(b[3].contents);
            for(unsigned i=0;i<n;++i){
                unsigned owner=0;std::memcpy(&owner,pairs+68*i+1,sizeof(owner));
                mapping[4*i+3]=i;cortex[4*i+3]=0;
                for(unsigned j=0;j<3;++j){
                    unsigned k=3*i+j;mapping[4*i+j]=k;cortex[4*i+j]=k;
                    for(unsigned a=0;a<3;++a){
                        mv[4*k+a]=pairs[68*i+4*(10+j)+a]+delta[16*i+4*j+a];
                        bv[4*k+a]=tris[40*i+4*j+a]+poses[8*owner+a];
                    }
                    mv[4*k+3]=0;bv[4*k+3]=0;
                }
            }
            [e setBuffer:map offset:0 atIndex:10];[e setBuffer:faces offset:0 atIndex:11];
            [e setBuffer:bone offset:0 atIndex:12];[e setBuffer:skin offset:0 atIndex:13];
        }
        [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted){NSLog(@"%@",cb.error);return 2;}
        auto pool=static_cast<const float*>(b[6].contents),gap=static_cast<const float*>(b[7].contents);
        std::cout<<std::setprecision(9);
        for(unsigned i=first;i<first+16*n;++i){std::cout<<gap[i];for(unsigned k=0;k<4;++k)std::cout<<' '<<pool[28*i+24+k];std::cout<<'\n';}
    }
}
