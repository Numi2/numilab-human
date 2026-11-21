#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "numi/matter/fem_compensated_deformation.h"

static void readExact(void* buffer, std::size_t bytes) {
    if (!std::cin.read(static_cast<char*>(buffer), bytes))
        throw std::runtime_error("Incomplete displacement fixture");
}

static double determinant(const std::array<double, 9>& f) {
    return f[0] * (f[4]*f[8] - f[5]*f[7])
         - f[1] * (f[3]*f[8] - f[5]*f[6])
         + f[2] * (f[3]*f[7] - f[4]*f[6]);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 2) throw std::runtime_error("Supply the pose probe Metal library");
            const auto device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("Metal device unavailable");
            NSError* error = nil;
            auto url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
            auto library = [device newLibraryWithURL:url error:&error];
            auto function = [library newFunctionWithName:@"check_pose"];
            auto pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
            if (!pipeline) throw std::runtime_error(error ? error.localizedDescription.UTF8String : "Missing check_pose");
            auto queue = [device newCommandQueue];
            std::uint32_t sizes[2];
            std::array<double, 9> rotation;
            readExact(sizes, sizeof(sizes));
            readExact(rotation.data(), sizeof(rotation));
            const auto nodes = sizes[0];
            const auto rows = sizes[1];
            if (!nodes || nodes > 1000000 || !rows || rows > 4000000)
                throw std::runtime_error("Fixture dimensions out of range");
            constexpr std::uint32_t batch = 65536;
            const auto allocate = [device](std::size_t bytes) {
                const auto buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!buffer) throw std::runtime_error("Metal buffer allocation failed");
                return buffer;
            };
            const auto displacements = allocate(std::size_t(nodes) * 6 * sizeof(float));
            const auto corners = allocate(batch * 4 * sizeof(std::uint32_t));
            const auto inverses = allocate(batch * 9 * sizeof(float));
            const auto output = allocate(batch * 18 * sizeof(float));
            readExact(displacements.contents, displacements.length);

            double minJ = std::numeric_limits<double>::infinity();
            double maxJ = 0, maxError = 0, maxHostDifference = 0;
            std::uint32_t nonpositive = 0;
            const float identity[9] = {1,0,0,0,1,0,0,0,1};
            for (std::uint32_t first = 0; first < rows; first += batch) {
                const std::uint32_t count = std::min(batch, rows - first);
                readExact(corners.contents, count * 4 * sizeof(std::uint32_t));
                readExact(inverses.contents, count * 9 * sizeof(float));
                const auto nodeIds = static_cast<const std::uint32_t*>(corners.contents);
                for (std::size_t i = 0; i < count * 4; ++i)
                    if (nodeIds[i] >= nodes) throw std::runtime_error("Fixture corner out of range");
                auto command = [queue commandBuffer];
                auto encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:displacements offset:0 atIndex:0];
                [encoder setBuffer:corners offset:0 atIndex:1];
                [encoder setBuffer:inverses offset:0 atIndex:2];
                [encoder setBuffer:output offset:0 atIndex:3];
                [encoder setBytes:&count length:sizeof(count) atIndex:4];
                [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error("Metal deformation dispatch failed");
                const auto deformations = static_cast<const float*>(output.contents);
                const auto displacement = static_cast<const float*>(displacements.contents);
                const auto inverse = static_cast<const float*>(inverses.contents);
                for (std::uint32_t i = 0; i < count; ++i) {
                    float uHigh[12], uLow[12], expectedHigh[9], expectedLow[9];
                    for (unsigned corner = 0; corner < 4; ++corner) {
                        const float* u = displacement + 6 * nodeIds[4*i+corner];
                        std::copy_n(u, 3, uHigh + 3*corner);
                        std::copy_n(u + 3, 3, uLow + 3*corner);
                    }
                    nmFEMReferenceDeformationWords(uHigh, uLow, inverse + 9*i, identity,
                                                   expectedHigh, expectedLow);
                    std::array<double, 9> f;
                    for (unsigned j = 0; j < f.size(); ++j) {
                        f[j] = double(deformations[18*i+j]) + deformations[18*i+9+j];
                        if (!std::isfinite(f[j]))
                            throw std::runtime_error("Nonfinite deformation component");
                        const double reference = double(expectedHigh[j]) + expectedLow[j];
                        maxError = std::max(maxError, std::abs(f[j] - rotation[j]));
                        maxHostDifference = std::max(maxHostDifference, std::abs(f[j] - reference));
                    }
                    const auto jacobian = determinant(f);
                    nonpositive += jacobian <= 0;
                    minJ = std::min(minJ, jacobian);
                    maxJ = std::max(maxJ, jacobian);
                }
            }
            if (std::cin.peek() != EOF) throw std::runtime_error("Trailing fixture bytes");
            std::cout << std::setprecision(17)
                      << "{\"device\":\"" << device.name.UTF8String << "\",\"cells\":" << rows
                      << ",\"minimum_J\":" << minJ << ",\"maximum_J\":" << maxJ
                      << ",\"nonpositive_J\":" << nonpositive
                      << ",\"maximum_rotation_error\":" << maxError
                      << ",\"maximum_host_deformation_difference\":" << maxHostDifference << "}\n";
            return nonpositive == 0 && maxError <= 1.1e-5 && maxHostDifference <= 1e-10 ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
