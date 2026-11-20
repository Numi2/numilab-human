#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

struct alignas(16) Vec4 { float x, y, z, w; };
struct Position { Vec4 high, low; };
static_assert(sizeof(Position) == 32);

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const bool prefix = argc == 3 && std::string_view(argv[2]) == "--prefix";
            if (argc != 2 && !prefix)
                throw std::runtime_error("Specify the pinned Matter metallib and optional --prefix");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            NSError* error = nil;
            NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            if (!library) throw std::runtime_error([[error description] UTF8String]);
            id<MTLFunction> function = [library newFunctionWithName:
                prefix ? @"numi_matter_metal::nm_contact_reference_rest_sweep_check" :
                         @"numi_matter_metal::nm_contact_reference_unresolved_sweep_check"];
            if (!function) throw std::runtime_error("The installed library lacks the sweep probe");
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error([[error description] UTF8String]);

            std::uint32_t count;
            if (!(std::cin >> count) || count == 0 || count > 16)
                throw std::runtime_error("Invalid witness count");
            std::vector<Position> points(8u*count);
            std::vector<Vec4> settings(count);
            for (std::uint32_t row = 0; row < count; ++row) {
                auto& setting = settings[row];
                if (!(std::cin >> setting.x >> setting.y))
                    throw std::runtime_error("Each witness requires clearance and feature kind");
                for (std::uint32_t i = 0; i < 8; ++i) {
                    auto& point = points[8u*row+i];
                    if (!(std::cin >> point.high.x >> point.high.y >> point.high.z
                                   >> point.low.x >> point.low.y >> point.low.z))
                        throw std::runtime_error("Incomplete compensated endpoint");
                }
            }
            if (prefix) {
                std::vector<Position> expanded;
                std::vector<Vec4> parameters;
                expanded.reserve(12u*count);
                parameters.reserve(2u*count);
                for (std::uint32_t row = 0; row < count; ++row) {
                    Vec4 axis{0,0,0,settings[row].y};
                    if (!(std::cin >> axis.x >> axis.y >> axis.z))
                        throw std::runtime_error("Each prefix witness requires the captured axis");
                    const auto first = points.begin()+8u*row;
                    expanded.insert(expanded.end(),first,first+4);
                    expanded.insert(expanded.end(),first,first+8);
                    parameters.push_back(axis);
                    parameters.push_back(settings[row]);
                }
                points = std::move(expanded);
                settings = std::move(parameters);
            }
            const auto input = [device](const auto& values) {
                return [device newBufferWithBytes:values.data()
                    length:values.size()*sizeof(values.front())
                    options:MTLResourceStorageModeShared];
            };
            const auto source = input(points);
            const auto configuration = input(settings);
            constexpr std::uint32_t canary = 0xd15c1ea4u;
            const std::uint32_t outputWords = prefix ? 2u*count : count;
            const auto output = input(std::vector<std::uint32_t>(outputWords+4u,canary));
            if (!source || !configuration || !output)
                throw std::runtime_error("Could not allocate probe buffers");

            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) throw std::runtime_error("Could not encode the Metal probe");
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:source offset:0 atIndex:0];
            [encoder setBuffer:configuration offset:0 atIndex:1];
            [encoder setBuffer:output offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(1,1,1)
                    threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error([[command.error description] UTF8String]);
            const auto* results = static_cast<const std::uint32_t*>(output.contents);
            for (std::uint32_t i = outputWords; i < outputWords+4u; ++i)
                if (results[i] != canary)
                    throw std::runtime_error("Sweep probe wrote beyond the record span");
            if (prefix) {
                const auto* values = static_cast<const float*>(output.contents);
                std::cout.precision(9);
                for (std::uint32_t i = 0; i < count; ++i)
                    std::cout << values[2u*i] << " " << values[2u*i+1u] << '\n';
            } else {
                for (std::uint32_t i = 0; i < count; ++i)
                    std::cout << "witness=" << i << " clearance_m=" << settings[i].x
                              << " full_sweep_separated=" << results[i] << '\n';
            }
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
