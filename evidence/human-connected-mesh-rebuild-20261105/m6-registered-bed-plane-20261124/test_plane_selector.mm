#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metalrobo/numi_human_support_geometry_gpu.h"
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <initializer_list>

int main(int argc,const char **argv) { @autoreleasepool {
    if(argc!=2)return 1;
    auto device=MTLCreateSystemDefaultDevice();NSError* err=nil;
    auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    if(!lib){NSLog(@"%@",err);return 2;}
    MRHumanRestingVertexMap map[2]{};
    MRHumanRestingInfluence influence[2]{};
    for(unsigned i=0;i<2;++i){
        map[i].firstInfluence=i;map[i].influenceCount=1;map[i].deformationKind=3;
        influence[i].positionAndWeight={i==0?1.f:-1.f,0.f,i==0?0.f:.001f,1.f};
    }
    MRHumanRestingSupportDispatchGPU dispatch{};dispatch.counts={2,1,1,1};dispatch.strides={1,0,0,0};dispatch.identity={0,2,1,3};
    dispatch.prediction.groundPointAndTimestep={0.f,0.f,.02f,.002f};
    MRArticulatedBodyPoseGPU body{};body.orientation={0.f,0.f,0.f,1.f};
    mr_float4 low{},orientation=body.orientation;unsigned region[2]{};
    auto buffer=[&](void* p,size_t bytes){return [device newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];};
    unsigned cases=0;
    for (bool predicted : {false,true}) for(bool tilt : {false,true}) {
        bool no=false,yes=true;
        MTLFunctionConstantValues* fc=[MTLFunctionConstantValues new];
        [fc setConstantValue:&yes type:MTLDataTypeBool atIndex:42];[fc setConstantValue:&no type:MTLDataTypeBool atIndex:15];
        [fc setConstantValue:&no type:MTLDataTypeBool atIndex:47];[fc setConstantValue:&predicted type:MTLDataTypeBool atIndex:48];
        auto fn=[lib newFunctionWithName:@"nm_human_registered_plane_positions" constantValues:fc error:&err];
        if(!fn){NSLog(@"%@",err);return 3;}
        auto pipeline=[device newComputePipelineStateWithFunction:fn error:&err];
        if(!pipeline){NSLog(@"%@",err);return 4;}
        unsigned init[2]={~0u,~0u},zero[2]={0,0};
        auto keys=buffer(init,sizeof(init)),minimum=buffer(init,4),invalid=buffer(zero,sizeof(zero));
        float x=tilt?.002f:0.f;mr_float4 plane{x,0.f,std::sqrt(1.f-x*x),-.02f};
        auto q=[device newCommandQueue];auto cmd=[q commandBuffer];auto e=[cmd computeCommandEncoder];
        [e setComputePipelineState:pipeline];[e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
        [e setBytes:map length:sizeof(map) atIndex:1];[e setBytes:influence length:sizeof(influence) atIndex:2];
        [e setBytes:&body length:sizeof(body) atIndex:3];[e setBytes:&low length:sizeof(low) atIndex:4];
        [e setBytes:region length:sizeof(region) atIndex:5];[e setBuffer:keys offset:0 atIndex:6];[e setBuffer:minimum offset:0 atIndex:7];[e setBuffer:invalid offset:0 atIndex:8];
        [e setBytes:&orientation length:sizeof(orientation) atIndex:9];
        MRHumanRestingSupportRotationZBasisGPU basis{};[e setBytes:&basis length:sizeof(basis) atIndex:10];
        mr_float4 velocities[4]{};for(auto& v:velocities){v.x=.1f;v.w=1.f;}if(predicted)[e setBytes:velocities length:sizeof(velocities) atIndex:13];
        [e setBytes:&plane length:sizeof(plane) atIndex:14];
        [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)];[e endEncoding];[cmd commit];[cmd waitUntilCompleted];
        if(cmd.error){NSLog(@"%@",cmd.error);return 5;}
        auto k=(unsigned*)keys.contents;
        for(unsigned i=0;i<2;++i){
            float value=std::bit_cast<float>(k[i]^(k[i]&0x80000000u?0x80000000u:0xffffffffu));
            auto p=influence[i].positionAndWeight;double expected=double(plane.x)*p.x+double(plane.z)*p.z+plane.w+(predicted?.002*plane.x*.1:0.);
            if(!std::isfinite(value)||std::abs(value-expected)>2.e-8){std::fprintf(stderr,"pred=%d tilt=%d row=%u key=%08x got=%.9g want=%.9g flags=%u,%u\n",predicted,tilt,i,k[i],value,expected,((unsigned*)invalid.contents)[0],((unsigned*)invalid.contents)[1]);return 6;}
        }
        if(((unsigned*)invalid.contents)[0]||((unsigned*)invalid.contents)[1]||*((unsigned*)minimum.contents)!=k[tilt?1:0])return 7;
        ++cases;
    }
    std::printf("{\"device\":\"%s\",\"cases\":%u,\"selected_flat\":0,\"selected_tilted\":1,\"current_and_predicted\":true}\n",device.name.UTF8String,cases);
    return 0;
}}
