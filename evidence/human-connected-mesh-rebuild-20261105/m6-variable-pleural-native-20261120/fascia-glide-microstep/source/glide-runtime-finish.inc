        if (glidePlan != nullptr) {
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            const float invDt = 1.0f/frameTimestep;
            [encoder setComputePipelineState:glidePlan->finish];
            [encoder setBytes:&glidePlan->dispatch length:sizeof(glidePlan->dispatch) atIndex:0u];
            [encoder setBytes:&invDt length:sizeof(invDt) atIndex:1u];
            [encoder setBuffer:glidePlan->samples offset:0u atIndex:2u];
            [encoder setBuffer:glidePlan->rowImpulses offset:0u atIndex:3u];
            [encoder setBuffer:glidePlan->wrenches offset:0u atIndex:4u];
            [encoder setBuffer:state.statuses offset:0u atIndex:5u];
            const auto count = glidePlan->dispatch.bodyPoseStride;
            const auto width = std::min<NSUInteger>(count, glidePlan->finish.threadExecutionWidth);
            [encoder dispatchThreads:MTLSizeMake(count,1u,1u)
                  threadsPerThreadgroup:MTLSizeMake(width,1u,1u)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
