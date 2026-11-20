#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 3)
                throw std::runtime_error("Pass the original and cached Matter libraries");
            constexpr unsigned environments = 2, nodes = 6, tets = 4;
            constexpr unsigned nodeCount = environments * nodes;
            constexpr unsigned tetCount = environments * tets;
            NMMatterDispatchGPU scene{};
            scene.environmentCount = environments;
            scene.femNodeCount = nodes;
            scene.tetrahedronCount = tets;
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) throw std::runtime_error("Metal command queue unavailable");

            auto makePipeline = [&](const char* path, NSString* name) {
                NSError* error = nil;
                id<MTLLibrary> library = [device newLibraryWithURL:
                    [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]]
                    error:&error];
                if (!library) throw std::runtime_error([[error description] UTF8String]);
                NSString* exported = [@"numi_matter_metal::" stringByAppendingString:name];
                id<MTLFunction> function = [library newFunctionWithName:exported];
                if (!function) throw std::runtime_error([exported UTF8String]);
                auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) throw std::runtime_error([[error description] UTF8String]);
                return pipeline;
            };
            auto oldElement = makePipeline(argv[1], @"nm_fgmres_precondition_patch_elements");
            auto oldGather = makePipeline(argv[1], @"nm_fgmres_precondition_patch_gather");
            auto build = makePipeline(argv[2], @"nm_fgmres_build_patch_linearization_cache");
            auto newElement = makePipeline(argv[2], @"nm_fgmres_precondition_cached_patch_elements");
            auto newGather = makePipeline(argv[2], @"nm_fgmres_precondition_cached_patch_gather");

            auto buffer = [&](auto const& value) {
                return [device newBufferWithBytes:value.data() length:sizeof(value[0])*value.size()
                        options:MTLResourceStorageModeShared];
            };
            constexpr auto unused = std::numeric_limits<unsigned>::max();
            const std::array<nm_uint4, tets> topology{{
                {0,1,2,3}, {1,2,3,4}, {0,2,3,5}, {2,3,4,unused}
            }};
            std::array<NMTetrahedronGPU, tetCount> geometry{};
            std::array<NMIncidenceRangeGPU, nodeCount> ranges{};
            std::vector<unsigned> incidence;
            for (unsigned e = 0; e < environments; ++e) {
                for (unsigned i = 0; i < tets; ++i)
                    geometry[e*tets+i].nodes = topology[i];
                for (unsigned node = 0; node < nodes; ++node) {
                    const auto first = static_cast<unsigned>(incidence.size());
                    for (unsigned t = 0; t < tets; ++t) {
                        const auto c = topology[t];
                        if (node == c.x || node == c.y || node == c.z || node == c.w)
                            incidence.push_back(t);
                    }
                    ranges[e*nodes+node] = {first, static_cast<unsigned>(incidence.size())-first,
                                              node == 5 ? unused : 0u, 0u};
                }
            }
            auto incidenceBuffer = buffer(incidence);
            auto rangeBuffer = buffer(ranges);
            id<MTLBuffer> cache = [device newBufferWithLength:32u*tetCount
                                      options:MTLResourceStorageModePrivate];
            if (!cache) throw std::runtime_error("No private patch cache");
            for (unsigned phase = 0; phase < 3; ++phase) {
                std::array<nm_float4, nodeCount> inverse{}, basis{}, output{};
                std::array<nm_float4, tetCount> corrections{};
                std::array<NMFGMRESStateGPU, environments> states{};
                states[1].diagnostics.z = phase == 0 ? 1.0f : 0.0f;
                for (unsigned i = 0; i < tetCount; ++i)
                    geometry[i].identity.w = ((i+phase) % 3u) ? NM_OBJECT_ACTIVE : 0u;
                for (unsigned i = 0; i < nodeCount; ++i) {
                    const float n = static_cast<float>(i + 1);
                    inverse[i] = {i%2 ? 0.0f : 0.7f*n, 2.0f/(n+phase),
                                  i%3 ? 0.2f*n : -0.01f, i%4 ? 0.05f*n : -0.0f};
                    basis[i] = {i%2 ? std::numeric_limits<float>::quiet_NaN() : 0.13f*n,
                                -0.17f*n, 1.0e-12f*n, 0.33f*n};
                    output[i] = {0.67f*n, -0.91f*n, 1.0e-6f*n, -0.72f*n};
                }
                auto tetBuffer = buffer(geometry), diagonalBuffer = buffer(inverse);
                auto basisBuffer = buffer(basis), stateBuffer = buffer(states);
                auto oldScratch = buffer(corrections), newScratch = buffer(corrections);
                auto oldOutput = buffer(output), newOutput = buffer(output);
                auto command = [queue commandBuffer];
                auto gpu = [command computeCommandEncoder];
                auto run = [&](id<MTLComputePipelineState> pipe, unsigned threads,
                               std::initializer_list<id<MTLBuffer>> args) {
                    [gpu setComputePipelineState:pipe];
                    [gpu setBytes:&scene length:sizeof(scene) atIndex:0];
                    unsigned index = 1;
                    for (auto b : args) {
                        if (b) [gpu setBuffer:b offset:0 atIndex:index];
                        ++index;
                    }
                    [gpu dispatchThreads:MTLSizeMake(threads,1,1)
                     threadsPerThreadgroup:MTLSizeMake(16,1,1)];
                    [gpu memoryBarrierWithScope:MTLBarrierScopeBuffers];
                };
                run(oldElement,tetCount,{tetBuffer,basisBuffer,diagonalBuffer,oldScratch,stateBuffer});
                run(oldGather,nodeCount,{tetBuffer,incidenceBuffer,rangeBuffer,oldScratch,
                                         diagonalBuffer,oldOutput,stateBuffer});
                run(build,tetCount,{tetBuffer,diagonalBuffer,cache,stateBuffer});
                run(newElement,tetCount,{cache,basisBuffer,nil,newScratch,stateBuffer});
                run(newGather,nodeCount,{cache,incidenceBuffer,rangeBuffer,newScratch,
                                         diagonalBuffer,newOutput,stateBuffer});
                [gpu endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error([[command.error description] UTF8String]);
                if (std::memcmp(oldScratch.contents,newScratch.contents,sizeof(corrections)) ||
                    std::memcmp(oldOutput.contents,newOutput.contents,sizeof(output)))
                    throw std::runtime_error("Patch round-trip failed bitwise equivalence");
            }
            std::cout << [[device name] UTF8String]
                      << " bitwise patch/gather parity: three restart states, mixed masks, "
                         "inactive elements, converged environment, NaN masking and invalid corner\n";
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
