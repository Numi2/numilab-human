#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include "numi/matter/shared.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 3, "usage: diagnostic production.metallib diagnostic.metallib");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device missing");
            auto library = [&](const char* path) {
                NSError* error = nil;
                id<MTLLibrary> value = [device newLibraryWithURL:[NSURL fileURLWithPath:
                    [NSString stringWithUTF8String:path]] error:&error];
                if (!value) std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
                require(value != nil, "library unavailable");
                return value;
            };
            auto pipeline = [&](id<MTLLibrary> lib, NSString* name) {
                NSError* error = nil;
                id<MTLFunction> function = [lib newFunctionWithName:name];
                require(function != nil, "kernel missing");
                id<MTLComputePipelineState> value = [device newComputePipelineStateWithFunction:function error:&error];
                require(value != nil, "pipeline failed");
                return value;
            };
            auto evaluate = pipeline(library(argv[1]), @"numi_matter_metal::nm_human_support_evaluate");
            auto update = pipeline(library(argv[2]), @"diagnostic_newton_update");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            auto buffer = [&](const void* bytes, NSUInteger count) {
                id<MTLBuffer> value = [device newBufferWithBytes:bytes length:count options:MTLResourceStorageModeShared];
                require(value != nil, "buffer allocation failed");
                return value;
            };
            struct Case { const char* name; float mass; unsigned rows; float warmWeightRatio; float dt; };
            const Case cases[] = {
                {"cold_single_97kg",97,1,0,0.0001f},
                {"weight_seed_single_97kg",97,1,1,0.0001f},
                {"double_weight_seed_single_97kg",97,1,2,0.0001f},
                {"cold_six_97kg",97,6,0,0.0001f},
                {"weight_seed_six_97kg",97,6,1,0.0001f},
                {"cold_single_1kg",1,1,0,0.0001f},
                {"cold_single_97kg_half_dt",97,1,0,0.00005f},
            };
            unsigned physicalFailures = 0;
            std::printf("{\"schema\":\"numi.support.loaded-history-diagnostic.v1\",\"cases\":[\n");
            for (unsigned index = 0; index < sizeof(cases)/sizeof(cases[0]); ++index) {
                const auto c = cases[index];
                NMMatterDispatchGPU dispatch{};
                dispatch.environmentCount = 1;
                dispatch.rigidGeneralizedCapacity = 1;
                NMHumanSupportDispatchGPU support{};
                support.contactCount = c.rows;
                support.articulatedNv = support.bodyCount = support.bodyStride = 1;
                support.groundNormal.y = 1;
                support.groundPointAndTimestep.w = c.dt;
                const float freeVelocity = -9.81f*c.dt;
                const float weightImpulse = c.mass*9.81f*c.dt;
                std::vector<NMHumanSupportContactGPU> contacts(c.rows);
                std::vector<nm_float4> accepted(c.rows), candidate(c.rows);
                std::vector<NMContactSampleGPU> samples(c.rows);
                std::vector<NMHumanSupportConsequenceGPU> consequences(c.rows);
                std::vector<float> jacobians(c.rows*3, 0);
                for (unsigned row = 0; row < c.rows; ++row) {
                    contacts[row].identity = {0, row, row, 0};
                    contacts[row].frictionSlopAndStabilization = {0,0.02f,0.2f,0};
                    accepted[row].w = weightImpulse*c.warmWeightRatio/c.rows;
                    jacobians[3*row+1] = 1;
                }
                MRBodyStateGPU body{};
                body.orientation.w = 1;
                body.position.y = c.dt*freeVelocity;
                body.linearVelocityAndInverseMass.y = freeVelocity;
                body.linearVelocityAndInverseMass.w = 1/c.mass;
                float generalized = 0;
                NMMatterStatusGPU status{};
                nm_float4 parameters{c.mass,c.dt,freeVelocity,0};
                auto contactsBuffer = buffer(contacts.data(),contacts.size()*sizeof(contacts[0]));
                auto bodyBuffer = buffer(&body,sizeof(body));
                auto generalizedBuffer = buffer(&generalized,sizeof(generalized));
                auto jacobianBuffer = buffer(jacobians.data(),jacobians.size()*sizeof(float));
                auto acceptedBuffer = buffer(accepted.data(),accepted.size()*sizeof(accepted[0]));
                auto candidateBuffer = buffer(candidate.data(),candidate.size()*sizeof(candidate[0]));
                auto sampleBuffer = buffer(samples.data(),samples.size()*sizeof(samples[0]));
                auto consequenceBuffer = buffer(consequences.data(),consequences.size()*sizeof(consequences[0]));
                auto statusBuffer = buffer(&status,sizeof(status));
                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil, "command allocation failed");
                for (unsigned iteration = 0; iteration <= 5; ++iteration) {
                    auto encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:evaluate];
                    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
                    [encoder setBytes:&support length:sizeof(support) atIndex:1];
                    [encoder setBuffer:contactsBuffer offset:0 atIndex:2];
                    [encoder setBuffer:bodyBuffer offset:0 atIndex:3];
                    [encoder setBuffer:generalizedBuffer offset:0 atIndex:4];
                    [encoder setBuffer:jacobianBuffer offset:0 atIndex:5];
                    [encoder setBuffer:acceptedBuffer offset:0 atIndex:6];
                    [encoder setBuffer:candidateBuffer offset:0 atIndex:7];
                    [encoder setBuffer:sampleBuffer offset:0 atIndex:8];
                    [encoder setBuffer:consequenceBuffer offset:0 atIndex:9];
                    [encoder setBuffer:statusBuffer offset:0 atIndex:10];
                    [encoder dispatchThreads:MTLSizeMake(c.rows,1,1) threadsPerThreadgroup:MTLSizeMake(c.rows,1,1)];
                    [encoder endEncoding];
                    if (iteration == 5) break;
                    encoder = [command computeCommandEncoder];
                    [encoder setComputePipelineState:update];
                    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
                    [encoder setBytes:&c.rows length:sizeof(c.rows) atIndex:1];
                    [encoder setBuffer:bodyBuffer offset:0 atIndex:2];
                    [encoder setBuffer:sampleBuffer offset:0 atIndex:3];
                    [encoder setBuffer:generalizedBuffer offset:0 atIndex:4];
                    [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                    [encoder endEncoding];
                }
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
                require(((NMMatterStatusGPU*)statusBuffer.contents)->code == 0, "production contact rejected diagnostic input");
                const auto* result = (const MRBodyStateGPU*)bodyBuffer.contents;
                const auto* sampled = (const NMContactSampleGPU*)sampleBuffer.contents;
                double impulse = 0;
                for (unsigned row=0; row<c.rows; ++row) impulse += sampled[row].impulseAndNormal.w;
                const double v = result->linearVelocityAndInverseMass.y;
                const double residual = c.mass*(freeVelocity-v)+impulse;
                const double analyticVelocity = (weightImpulse*c.warmWeightRatio-weightImpulse)/
                    (c.mass + c.rows*(c.warmWeightRatio<1 ? 1.2 : 1.0));
                require(std::abs(v-analyticVelocity)<2e-9, "production result disagrees with frozen-history scalar law");
                require(std::abs(residual)<3e-8, "scalar momentum root failed to converge");
                const bool physicalPass = std::abs(v)<1e-7 && std::abs(impulse-weightImpulse)<3e-8;
                physicalFailures += !physicalPass;
                std::printf("%s{\"name\":\"%s\",\"mass_kg\":%.9g,\"rows\":%u,\"warm_weight_ratio\":%.9g,\"dt_s\":%.9g,\"velocity_m_s\":%.12g,\"gap_m\":%.12g,\"normal_impulse_Ns\":%.12g,\"required_weight_impulse_Ns\":%.12g,\"support_fraction\":%.12g,\"momentum_residual_Ns\":%.12g,\"normal_complementarity_Ns_m_s\":%.12g,\"hard_support_pass\":%s,\"gpu_seconds\":%.9g}",
                    index ? ",\n" : "", c.name,c.mass,c.rows,c.warmWeightRatio,c.dt,v,result->position.y,impulse,weightImpulse,
                    impulse/weightImpulse,residual,impulse*v,physicalPass?"true":"false",command.GPUEndTime-command.GPUStartTime);
            }
            std::printf("\n],\"physical_failures\":%u,\"hard_support_qualified\":%s}\n",physicalFailures,physicalFailures?"false":"true");
            return physicalFailures ? 2 : 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr,"diagnostic execution failure: %s\n",error.what());
            return 1;
        }
    }
}
