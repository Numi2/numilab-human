            if (glidePlan != nullptr) {
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [encoder setComputePipelineState:glidePlan->refresh];
                [encoder setBytes:&state.dispatch length:sizeof(state.dispatch) atIndex:0u];
                [encoder setBytes:&micro length:sizeof(micro) atIndex:1u];
                [encoder setBytes:&glidePlan->dispatch length:sizeof(glidePlan->dispatch) atIndex:2u];
                [encoder setBuffer:glidePlan->samples offset:0u atIndex:3u];
                [encoder setBuffer:state.femAccepted offset:0u atIndex:4u];
                [encoder setBuffer:glidePlan->poses offset:0u atIndex:5u];
                [encoder setBuffer:femExternalForces offset:0u atIndex:6u];
                [encoder setBuffer:glidePlan->rowImpulses offset:0u atIndex:7u];
                [encoder setBuffer:state.femNodeRanges offset:0u atIndex:8u];
                [encoder setBuffer:state.objects offset:0u atIndex:9u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:10u];
                [encoder setBuffer:state.adaptive offset:0u atIndex:11u];
                [encoder setBuffer:state.statuses offset:0u atIndex:12u];
                [encoder setBuffer:glidePlan->previousForce offset:0u atIndex:13u];
                const auto count = glidePlan->dispatch.femBodyContactSampleCount;
                const auto width = std::min<NSUInteger>(count, glidePlan->refresh.threadExecutionWidth);
                [encoder dispatchThreads:MTLSizeMake(count,1u,1u)
                      threadsPerThreadgroup:MTLSizeMake(width,1u,1u)];
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
