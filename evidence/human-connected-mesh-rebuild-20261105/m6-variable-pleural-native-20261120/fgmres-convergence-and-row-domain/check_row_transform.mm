#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/shared.h"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 2) throw std::runtime_error("Pass the built Matter library");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [device newCommandQueue];
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                error:&error];
            if (!queue || !library) throw std::runtime_error("Matter Metal library unavailable");
            auto pipeline = [&](NSString* name) {
                NSError* pipelineError = nil;
                NSString* fullName = [@"numi_matter_metal::" stringByAppendingString:name];
                id<MTLFunction> function = [library newFunctionWithName:fullName];
                auto result = [device newComputePipelineStateWithFunction:function
                                                             error:&pipelineError];
                if (!result) throw std::runtime_error([fullName UTF8String]);
                return result;
            };
            const auto scaleRows = pipeline(@"nm_fgmres_apply_left_row_equilibration");
            const auto unscaleRows = pipeline(@"nm_fgmres_unscale_preconditioner_input");
            auto buffer = [&](const auto& data) {
                auto result = [device newBufferWithBytes:data.data()
                        length:data.size() * sizeof(data[0]) options:MTLResourceStorageModeShared];
                if (!result) throw std::runtime_error("Fixture buffer allocation failed");
                return result;
            };

            // A nonaligned mechanical head, cases with and without nonmechanical
            // rows, and a tail spanning several strides.
            constexpr unsigned nodes = 259;
            const std::array<unsigned, 3> totals{nodes, 2 * nodes + 5, 3 * nodes + 73};
            NMMatterDispatchGPU dispatch{};
            dispatch.environmentCount = 1;
            dispatch.femNodeCount = nodes;
            std::vector<float> scaling(nodes);
            for (unsigned i = 0; i < nodes; ++i) scaling[i] = 0.5f + float(i % 31) / 7.0f;
            auto scales = buffer(scaling);
            const std::array<unsigned, 4> active{2, 1, 1, 0};
            const std::array<unsigned, 4> inactive{0, 1, 1, 0};
            auto activeRange = buffer(active);
            auto inactiveRange = buffer(inactive);
            for (unsigned total : totals) {
                NMFGMRESLayoutGPU layout{};
                layout.unknownCount = total;
                std::vector<nm_float4> values(total), sentinel(total, nm_float4{-9,-8,-7,-6});
                for (unsigned row = 0; row < total; ++row) {
                    const float n = float(row + 1);
                    values[row] = {n / 13.0f, -n / 29.0f, n * 1.0e-11f,
                                   row % 2 ? -0.0f : n / 23.0f};
                }
                auto original = buffer(values);
                auto regular = buffer(sentinel), strided = buffer(sentinel), stopped = buffer(sentinel);
                auto regularAliased = buffer(values), boundedAliased = buffer(values);
                auto command = [queue commandBuffer];
                auto encoder = [command computeCommandEncoder];
                auto encode = [&](id<MTLComputePipelineState> pipe,
                    id<MTLBuffer> source, id<MTLBuffer> output, unsigned mode,
                    id<MTLBuffer> range) {
                    [encoder setComputePipelineState:pipe];
                    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
                    [encoder setBuffer:scales offset:0 atIndex:1];
                    [encoder setBuffer:source offset:0 atIndex:2];
                    [encoder setBuffer:output offset:0 atIndex:3];
                    if (pipe == unscaleRows)
                        [encoder setBytes:&mode length:sizeof(mode) atIndex:4];
                    [encoder setBytes:&layout length:sizeof(layout) atIndex:30];
                    if (range) {
                        [encoder useResource:range usage:MTLResourceUsageRead];
                        [encoder dispatchThreadgroupsWithIndirectBuffer:range
                            indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                    } else {
                        [encoder dispatchThreads:MTLSizeMake(total,1,1)
                            threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                    }
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                };
                encode(unscaleRows, original, regular, 0, nil);
                encode(unscaleRows, original, strided, 1, activeRange);
                encode(unscaleRows, original, stopped, 1, inactiveRange);
                encode(scaleRows, regularAliased, regularAliased, 0, nil);
                encode(scaleRows, boundedAliased, boundedAliased, 0, activeRange);
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error([[command.error description] UTF8String]);
                const auto bytes = total * sizeof(nm_float4);
                if (std::memcmp(regular.contents, strided.contents, bytes) ||
                    std::memcmp(stopped.contents, sentinel.data(), bytes) ||
                    std::memcmp(regularAliased.contents, boundedAliased.contents, bytes))
                    throw std::runtime_error("Row-transform parity failed");
            }
            std::cout << "row_transform_gpu_parity=pass cases=" << totals.size()
                      << " complete_vector_bytes_equal=true indirect_empty_untouched=true"
                      << " gpu=" << [[device name] UTF8String] << '\n';
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }
}
