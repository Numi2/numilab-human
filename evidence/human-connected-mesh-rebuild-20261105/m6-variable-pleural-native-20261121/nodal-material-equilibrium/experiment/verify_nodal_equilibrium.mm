#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
int main(int argc,char** argv){@autoreleasepool{
    if(argc!=4)return 2;
    auto device=MTLCreateSystemDefaultDevice();NSError* error=nil;
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    auto pipe=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_pelvic_nodal_equilibrium"] error:&error];
    check(pipe!=nil,"material pipeline");
    const std::filesystem::path assets=argv[2],run=argv[3];
    auto make=[&](size_t n){auto b=[device newBufferWithLength:n options:MTLResourceStorageModeShared];check(b!=nil,"allocation");std::memset(b.contents,0,n);return b;};
    auto bytes=[&](std::filesystem::path path){NSData* v=[NSData dataWithContentsOfFile:@(path.c_str())];check(v.length>0,"fixture file");return v;};
    auto load=[&](const char* n){auto v=bytes(assets/n);auto b=make(v.length);std::memcpy(b.contents,v.bytes,v.length);return b;};
    auto sample=bytes(run/"candidate-force-nodes.bin"),ports=bytes(run/"candidate-force-ports.bin");
    const unsigned n=static_cast<unsigned>(sample.length/32);
    check(32*n==sample.length&&ports.length==512*64,"capture layout");
    std::array<id<MTLBuffer>,13> b{make(8),make(32),load("free-node-rows.bin"),make(8*n*4),
        load("pelvic-projector-rows.bin"),make(n*16),make(n*16),load("pelvic-cell-incidence.bin"),
        load("pelvic-cell-corners.bin"),load("material-cells.bin"),load("pelvic-reference-inverse.bin"),
        load("pelvic-cell-volume.bin"),make(n*16)};
    ((unsigned*)b[0].contents)[0]=((unsigned*)b[0].contents)[1]=1;
    const auto* port=static_cast<const float*>(ports.bytes);const auto* nodes=static_cast<const float*>(sample.bytes);
    NSMutableData* docBytes=[NSMutableData dataWithData:bytes(run/"nodal-pressure-bound.json")];
    NSDictionary* audit=[NSJSONSerialization JSONObjectWithData:docBytes options:0 error:&error];
    check(audit!=nil,"witness metadata");
    for(unsigned s=0;s<2;++s){
        NSDictionary* d=audit[@"sides"][@(s?"1":"0")];
        ((float*)b[1].contents)[4*s]=[d[@"native_residual_N"] floatValue];
        ((float*)b[1].contents)[4*s+1]=[d[@"native_scale_N"] floatValue];
    }
    for(unsigned i=0;i<n;++i){
        std::memcpy(static_cast<float*>(b[5].contents)+4*i,nodes+8*i,16);
        std::memcpy(static_cast<float*>(b[6].contents)+4*i,nodes+8*i+4,16);
    }
    auto mask=static_cast<unsigned*>(b[3].contents);
    for(unsigned k=0;k<512;++k){
        const float* rec=port+16*k;
        if(rec[3]<=0 || rec[4]<0)continue;
        for(unsigned j=0;j<4;++j)if(rec[12+j]!=0){
            unsigned i=static_cast<unsigned>(rec[8+j]);check(i<n,"contact node");
            mask[8*i+(k%256)/32]|=1u<<(k%32);
        }
    }
    auto queue=[device newCommandQueue];auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    [e setComputePipelineState:pipe];for(unsigned i=0;i<b.size();++i)[e setBuffer:b[i] offset:0 atIndex:i];
    [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e endEncoding];[cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"GPU execution");
    NSDictionary* parity=[NSJSONSerialization JSONObjectWithData:bytes(run/"local-block-parity.json") options:0 error:&error];
    NSDictionary* left=parity[@"1"];unsigned expected=[left[@"node"] unsignedIntValue];unsigned changed=0;
    double difference=0,mag=0;const auto* step=static_cast<const float*>(b[12].contents);
    for(unsigned i=0;i<n;++i){
        double len=0;for(unsigned k=0;k<3;++k)len+=step[4*i+k]*step[4*i+k];if(!len)continue;
        check(i==expected,"changed a different coordinate");++changed;
        for(unsigned k=0;k<3;++k){double correct=[left[@"correction_m"][k] doubleValue];double v=step[4*i+k]-correct;difference+=v*v;mag+=correct*correct;}
    }
    check(changed==1 && std::sqrt(difference)<2e-9,"material block parity");
    std::cout<<"changed_nodes,node,correction_error_m,relative_error\n"<<changed<<','<<expected<<','<<std::sqrt(difference)<<','<<std::sqrt(difference/mag)<<'\n';return 0;
}}
