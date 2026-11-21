#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/numi_human_shared.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool pass, const char* message) {
    if (pass) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

int main(int argc, const char** argv) {
    @autoreleasepool {
        require(argc == 3, "control and candidate metallib paths required");
        NSError* error = nil;
        id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
        require(gpu != nil, "Apple GPU unavailable");
        id<MTLCommandQueue> queue = [gpu newCommandQueue];
        require(queue != nil, "queue unavailable");
        auto pipeline = [&](const char* path, const char* entry) {
            auto library = [gpu newLibraryWithURL:[NSURL fileURLWithPath:@(path)] error:&error];
            require(library != nil, "load metallib");
            auto function = [library newFunctionWithName:
                [NSString stringWithFormat:@"numi_matter_metal::%s",entry]];
            require(function != nil, "load Metal entry point");
            auto state = [gpu newComputePipelineStateWithFunction:function error:&error];
            require(state != nil, "build compute pipeline");
            return state;
        };
        auto control = pipeline(argv[1], "nm_numi_human_assemble_fem_body_contact_loads_grouped_cached");
        auto candidate = pipeline(argv[2], "nm_numi_human_assemble_fem_body_contact_loads_slave_ranges_cached");
        auto buffer = [&](std::size_t bytes) {
            auto value = [gpu newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            require(value != nil, "allocate Metal input");
            std::memset(value.contents,0,bytes);
            return value;
        };

        constexpr uint32_t bodies = 86, multiSlaveCount = 179649, singleSlaveCount = 79;
        constexpr uint32_t groups = multiSlaveCount + singleSlaveCount;
        constexpr uint32_t rows = multiSlaveCount * bodies + singleSlaveCount;
        constexpr uint32_t alignedMapEnd = (32 + 8 * rows + 15) & ~15u;
        constexpr uint32_t seedsBase = alignedMapEnd + 64 * groups;
        constexpr uint32_t end = seedsBase + 96 * bodies;
        // This fixture uses the existing private packed contact storage so it
        // measures the same sample decode path as the retained fine scene.
        // The 86-body/79-singleton partition is representative, not a replay.
        auto samples = buffer(end);
        const uint32_t header[8]{rows,groups,bodies,0x4e484331u,32u,alignedMapEnd,seedsBase,end};
        std::memcpy(samples.contents, header, sizeof(header));
        auto* ids = static_cast<uint32_t*>(samples.contents)+8;
        auto* payload = static_cast<uint8_t*>(samples.contents);
        auto* templates = reinterpret_cast<NMNumiHumanFEMBodyContactSampleGPU*>(payload+alignedMapEnd);
        auto dense = buffer(rows * sizeof(uint32_t));
        auto compact = buffer(groups * sizeof(NMIncidenceRangeGPU));
        auto* denseSpans = static_cast<uint32_t*>(dense.contents);
        auto* spans = static_cast<NMIncidenceRangeGPU*>(compact.contents);
        uint32_t first = 0;
        for (uint32_t slave = 0; slave < groups; ++slave) {
            templates[slave].slaveNode = slave;
            const uint32_t count = slave < multiSlaveCount ? bodies : 1u;
            denseSpans[first] = count;
            spans[slave] = {first,count,0,0};
            for (uint32_t body = 0; body < count; ++body) {
                ids[2*(first+body)] = slave;
                ids[2*(first+body)+1] = body;
            }
            first += count;
        }
        require(first == rows, "unfinished sample inventory");
        for (uint32_t body=0; body<bodies; ++body)
            reinterpret_cast<NMNumiHumanFEMBodyContactSampleGPU*>(payload+seedsBase+96*body)->bodyIndex = body;
        auto cache = buffer(rows*sizeof(nm_float4));
        auto* forces = static_cast<nm_float4*>(cache.contents);
        std::vector<nm_float4> expected(groups, {0.125f,-0.25f,0.5f,0.0f});
        for (uint32_t row=0; row<rows; ++row) {
            if (row % 7919u != 0u && row > 1996u) continue;
            // Consecutive and scattered closures exercise summation order.
            const nm_float4 f{(row%2u ? -1.f : 1.f)*0.00025f, 0.000125f, 0.f, 0.0005f};
            forces[row]=f;
        }
        const uint32_t invalid = 83u;
        forces[invalid] = {NAN,0,0,0};
        for (uint32_t group=1u; group<groups; ++group) {
            nm_float4 subtotal{};
            for (uint32_t i=spans[group].first; i<spans[group].first+spans[group].count; ++i) {
                subtotal.x += forces[i].x;
                subtotal.y += forces[i].y;
                subtotal.z += forces[i].z;
            }
            expected[group].x += subtotal.x;
            expected[group].y += subtotal.y;
            expected[group].z += subtotal.z;
        }
        auto unused = buffer(1024);
        auto output = buffer(groups*sizeof(nm_float4));
        NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
        dispatch.abiVersion = NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;
        dispatch.environmentCount = 1u;
        dispatch.femNodeCount = groups;
        dispatch.femBodyContactSampleCount = rows;
        dispatch.bodyPoseStride = bodies;
        std::vector<nm_float4> initial(groups, {0.125f,-0.25f,0.5f,0.0f});
        const auto run = [&](bool ranges) {
            std::memcpy(output.contents,initial.data(),output.length);
            auto command = [queue commandBuffer];
            auto encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:ranges ? candidate : control];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
            [encoder setBuffer:samples offset:0 atIndex:1];
            [encoder setBuffer:unused offset:0 atIndex:2];
            [encoder setBuffer:unused offset:0 atIndex:3];
            [encoder setBuffer:output offset:0 atIndex:4];
            [encoder setBuffer:unused offset:0 atIndex:5];
            [encoder setBuffer:unused offset:0 atIndex:6];
            [encoder setBuffer:unused offset:0 atIndex:8];
            [encoder setBuffer:ranges ? compact : dense offset:0 atIndex:9];
            [encoder setBuffer:cache offset:0 atIndex:10];
            if (ranges) [encoder setBytes:&groups length:sizeof(groups) atIndex:11];
            [encoder dispatchThreads:MTLSizeMake(ranges ? groups : rows,1,1)
                    threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,"force kernel failed");
            require(std::memcmp(output.contents,expected.data(),output.length)==0,
                    "force output lost bitwise parity");
            return 1000.*(command.GPUEndTime-command.GPUStartTime);
        };
        require(run(false)>0. && run(true)>0.,"zero GPU timing");
        for (unsigned pair=0; pair<5; ++pair) {
            const bool firstRanges = (pair&1u)!=0;
            const double a = run(firstRanges), b = run(!firstRanges);
            std::printf("{\"pair\":%u,\"rows\":%u,\"ranges\":%u,\"control_gpu_ms\":%.9g,"
                        "\"candidate_gpu_ms\":%.9g,\"dense_dispatch_bytes\":%lu,\"range_dispatch_bytes\":%lu,"
                        "\"same_float_bytes\":%lu}\n", pair, rows, groups, firstRanges?b:a, firstRanges?a:b,
                        (unsigned long)dense.length,(unsigned long)compact.length,(unsigned long)output.length);
        }
    }
}
