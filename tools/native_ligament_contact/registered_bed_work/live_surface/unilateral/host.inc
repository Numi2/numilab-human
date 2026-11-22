        const auto encodeRegisteredBedResidual = [&](const uint32_t finalPass) {
            if (!state.unilateralBed) return;
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            dispatchThreads("nm_fem_registered_bed_residual", pairTotal, [&] {
                setDispatch();
                bindPrimalContactArguments(1u);
                [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.femResidual offset:0u atIndex:3u];
                [encoder setBuffer:state.bedDualLinearization offset:0u atIndex:4u];
                [encoder setBuffer:state.contactSamples offset:0u atIndex:5u];
                [encoder setBuffer:state.materials offset:0u atIndex:6u];
                [encoder setBuffer:state.objects offset:0u atIndex:7u];
                [encoder setBuffer:state.schedulers offset:0u atIndex:8u];
                [encoder setBuffer:state.mixedSolver offset:0u atIndex:9u];
                [encoder setBuffer:state.statuses offset:0u atIndex:10u];
                [encoder setBytes:&finalPass length:sizeof(finalPass) atIndex:11u];
                [encoder setBuffer:state.contactHistoriesCandidate offset:0u atIndex:12u];
            });
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };

        const auto encodeRegisteredBedOperator = [&](id<MTLBuffer> direction, NSUInteger directionOffset,
                id<MTLBuffer> work, id<MTLBuffer> states, uint32_t forceComplete) {
            if (!state.unilateralBed) return;
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            dispatchThreads("nm_fem_registered_bed_operator", pairTotal, [&] {
                setDispatch();
                bindPrimalContactArguments(1u);
                [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                [encoder setBuffer:state.bedDualLinearization offset:0u atIndex:3u];
                [encoder setBuffer:direction offset:directionOffset atIndex:4u];
                [encoder setBuffer:work offset:0u atIndex:5u];
                [encoder setBuffer:states offset:0u atIndex:6u];
                [encoder setBytes:&forceComplete length:sizeof(forceComplete) atIndex:7u];
            });
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
