#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/matter/human_respiration.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

int main(int argc, const char* argv[]) {
    if (argc != 2) return 1;
    @autoreleasepool {
        constexpr unsigned rowCount = 21;
        constexpr std::uint32_t thoracicMask = 2080819;
        constexpr unsigned abdominalRow = 6;
        static_assert((thoracicMask & (1u << abdominalRow)) == 0);
        const std::array<float, 2> timeAndRamp{0.002f, 5.0f};
        const auto url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        const auto device = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        const auto library = [device newLibraryWithURL:url error:&error];
        const auto function = [library newFunctionWithName:@"nm_human_stage_thoracic_pressure"];
        const auto pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        const auto queue = [device newCommandQueue];
        if (!queue || !pipeline) {
            std::cerr << [[error description] UTF8String] << '\n';
            return 2;
        }
        std::array<NMVascularCompartmentGPU, rowCount> baseline{};
        for (unsigned i = 0; i < rowCount; ++i) {
            baseline[i].identity.x = i + 1;
            baseline[i].compliance.w = thoracicMask & (1u << i) ? -500.0f : 20.0f + i;
        }
        const auto reference = [device newBufferWithBytes:baseline.data()
            length:sizeof(baseline) options:MTLResourceStorageModeShared];
        const auto trial = [device newBufferWithLength:sizeof(baseline) options:MTLResourceStorageModeShared];
        const auto accepted = [device newBufferWithLength:sizeof(NMHumanRespirationState)
            options:MTLResourceStorageModeShared];
        if (!reference || !trial || !accepted) return 3;

        struct Sample { unsigned step; float pressure; float expectedThoracic; };
        const std::array samples{
            Sample{0, -800.0f, -500.0f},
            Sample{1250, -800.0f, -650.0f},
            Sample{2500, -800.0f, -800.0f},
            Sample{3000, -380.0f, -380.0f},
        };
        for (const auto& sample : samples) {
            NMHumanRespirationState state{};
            state.status.x = sample.step;
            state.mechanics.z = sample.pressure;
            std::memcpy(accepted.contents, &state, sizeof(state));
            std::memset(trial.contents, 0xA5, sizeof(baseline));
            const auto command = [queue commandBuffer];
            const auto encoder = [command computeCommandEncoder];
            if (!command || !encoder) return 4;
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:reference offset:0 atIndex:0];
            [encoder setBuffer:trial offset:0 atIndex:1];
            [encoder setBuffer:accepted offset:0 atIndex:2];
            [encoder setBytes:timeAndRamp.data() length:sizeof(timeAndRamp) atIndex:3];
            [encoder setBytes:&thoracicMask length:sizeof(thoracicMask) atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(rowCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(rowCount, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) return 5;
            if (std::memcmp(reference.contents, baseline.data(), sizeof(baseline)) != 0 ||
                std::memcmp(accepted.contents, &state, sizeof(state)) != 0) return 6;
            const auto* rows = static_cast<const NMVascularCompartmentGPU*>(trial.contents);
            for (unsigned i = 0; i < rowCount; ++i) {
                const auto& output = rows[i];
                auto expected = baseline[i];
                if (thoracicMask & (1u << i)) {
                    if (std::abs(output.compliance.w - sample.expectedThoracic) > 1.0e-4f) return 7;
                    expected.compliance.w = output.compliance.w;
                }
                if (std::memcmp(&output, &expected, sizeof(output)) != 0) return 8;
            }
        }
        std::cout << device.name.UTF8String
                  << " pressure stage PASS: 21 rows, 4 accepted pressure samples, abdominal row byte-identical\n";
    }
}
