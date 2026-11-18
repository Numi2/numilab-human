#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class T>
static id<MTLBuffer> storage(id<MTLDevice> device, const std::vector<T>& items) {
    id<MTLBuffer> buffer = [device newBufferWithBytes:items.data()
        length:items.size()*sizeof(T) options:MTLResourceStorageModeShared];
    require(buffer != nil, "buffer allocation failed");
    return buffer;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc==2, "provide the exact retained Matter metallib");
            constexpr uint32_t count=8192, nodesPerPatch=8, width=24;
            constexpr uint32_t nodeCount=count*nodesPerPatch;
            constexpr uint32_t matrixWords=count*width*width;
            constexpr uint32_t pivotWords=count*width*3;
            auto device=MTLCreateSystemDefaultDevice();
            require(device != nil, "no Metal device");
            NSError* error=nil;
            auto library=[device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
            require(library != nil, "could not load retained metallib");
            auto queue=[device newCommandQueue];
            auto pipeline=[&](NSString* name) {
                auto function=[library newFunctionWithName:name];
                require(function != nil, "could not load graph kernel");
                NSError* pipelineError=nil;
                auto state=[device newComputePipelineStateWithFunction:function error:&pipelineError];
                require(state != nil, "could not create graph pipeline");
                return state;
            };
            const auto factor=pipeline(@"numi_matter_metal::nm_fgmres_graph_factor_patches");
            const auto solve=pipeline(@"numi_matter_metal::nm_fgmres_graph_solve_patches_capacity48_threads32");
            NMMatterDispatchGPU params{};
            params.abiVersion=NM_MATTER_ABI_VERSION;
            params.environmentCount=1u;
            params.femNodeCount=nodeCount;
            auto descriptor=storage(device,std::vector{params});
            std::vector<nm_uint4> rows(count);
            std::vector<uint32_t> nodes(nodeCount);
            std::vector<nm_float4> rhs(nodeCount);
            std::vector<float> raw(matrixWords);
            for (uint32_t patch=0; patch<count; ++patch) {
                rows[patch]={patch*nodesPerPatch,nodesPerPatch,patch*width*width,patch*width*3};
                for(uint32_t local=0;local<nodesPerPatch;++local) {
                    const uint32_t node=patch*nodesPerPatch+local;
                    nodes[node]=node;
                    rhs[node]={std::sin(node*.05f),std::cos(node*.03f),std::sin(node*.01f),0.0f};
                }
                for (uint32_t row=0;row<width;++row) {
                    for (uint32_t col=0;col<width;++col) {
                        const auto i=rows[patch].z+row*width+col;
                        const float cross=0.007f*std::cos(float(patch%256+row*31+col*11));
                        raw[i]=(row==col?2.0f+float(row)*0.04f:cross);
                    }
                }
            }
            const auto patchRows=storage(device,rows);
            const auto patchNodes=storage(device,nodes);
            const auto matrices=storage(device,raw);
            const auto original=storage(device,std::vector<float>(matrixWords));
            const auto pivots=storage(device,std::vector<uint32_t>(pivotWords));
            const auto validity=storage(device,std::vector<uint32_t>(count));
            const auto status=storage(device,std::vector<NMMatterStatusGPU>(1));
            const auto states=storage(device,std::vector<NMFGMRESStateGPU>(1));
            auto scalar=[&](uint32_t v){return storage(device,std::vector{v});};
            const auto zero=scalar(0u), one=scalar(1u), all=scalar(NM_INVALID_INDEX);
            const auto patches=scalar(count), nodeWords=scalar(nodeCount);
            const auto matrixCount=scalar(matrixWords), pivotCount=scalar(pivotWords);
            auto run=[&](id<MTLComputePipelineState> state,
                    const std::vector<id<MTLBuffer>>& buffers, uint32_t threads) {
                auto cb=[queue commandBuffer];
                auto enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:state];
                for(NSUInteger i=0;i<buffers.size();++i)
                    [enc setBuffer:buffers[i] offset:0 atIndex:i];
                [enc dispatchThreadgroups:MTLSizeMake(count,1,1)
                    threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                require(cb.status==MTLCommandBufferStatusCompleted,"graph dispatch failed");
                require(static_cast<const NMMatterStatusGPU*>(status.contents)->code==NM_STATUS_SUCCESS,
                    "graph kernel reported a failure");
                return (cb.GPUEndTime-cb.GPUStartTime)*1000.0;
            };
            run(factor,{descriptor,patchRows,matrices,pivots,validity,status,patches,
                matrixCount,pivotCount,original,states,zero,zero,one,zero,zero,all},256);
            const auto* valid=static_cast<const uint32_t*>(validity.contents);
            require(std::all_of(valid,valid+count,[](uint32_t v){return v==1;}),
                "some factors were invalid");
            require(std::memcmp(original.contents,raw.data(),raw.size()*sizeof(float))==0,
                "factor did not capture reference matrix bytes");
            const auto basis=storage(device,rhs);
            const auto candidate=storage(device,std::vector<NMFEMNodeStateGPU>(nodeCount));
            const auto r=storage(device,std::vector<nm_float4>(nodeCount));
            std::array<id<MTLBuffer>,2> corrections{
                storage(device,std::vector<nm_float4>(nodeCount)),
                storage(device,std::vector<nm_float4>(nodeCount))};
            const auto jacobi=storage(device,std::vector<uint32_t>(9));
            const std::array<id<MTLBuffer>,2> modes{one,scalar(9)};
            auto apply=[&](int variant){
                return run(solve,{descriptor,patchRows,patchNodes,basis,r,r,zero,matrices,
                    pivots,validity,corrections[variant],status,patches,nodeWords,
                    matrixCount,pivotCount,zero,states,original,modes[variant],
                    candidate,zero,zero,zero,zero,zero,jacobi,zero},32);
            };
            apply(0);apply(1);
            require(std::memcmp(corrections[0].contents,corrections[1].contents,
                nodeCount*sizeof(nm_float4))==0,"precomputing products changed solution bytes");
            const auto* result=static_cast<const nm_float4*>(corrections[0].contents);
            double sum=0.0;
            for(uint32_t i=0;i<nodeCount;++i) sum+=std::abs(result[i].x);
            require(sum>0.0 && std::isfinite(sum),"graph solve produced empty or invalid output");
            std::cout << "device=" << device.name.UTF8String << " patches=" << count
                << " dofs_per_patch=" << width << " byte_parity=1 abs_x_sum=" << sum << '\n';
            for(int pair=0;pair<12;++pair) {
                double timings[2];
                const int first=pair%2;
                timings[first]=apply(first);
                timings[1-first]=apply(1-first);
                std::cout<<std::setprecision(9)<<"pair="<<pair
                    <<" reference_ms="<<timings[0]
                    <<" candidate_ms="<<timings[1]<<'\n';
            }
        } catch(const std::exception& failure) {
            std::cerr<<failure.what()<<'\n';
            return 1;
        }
    }
}
