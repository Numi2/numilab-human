            if (state.releaseVolumeMode==2u && state.unilateralBed && state.femParallelConstitutiveBacktrackingEnabled) {
                // FEM previously evaluated dyadic fractions. Bed CCD and
                // vascular merit can select a different represented fraction.
                // Verify it before changing any candidate state. A reduced
                // fraction is submitted to the same support and vascular owners.
                const uint32_t budget=std::min<uint32_t>(state.mixedSolverValue.nonlinearIterations.w,NM_MIXED_LINE_SEARCH_MAX_STEPS);
                auto scratch=state.registeredBoundArguments;
                for(uint32_t pass=0;pass<4u;++pass){
                    dispatchThreads("nm_fem_registered_release_bound_begin",1u,[&] {
                        setDispatch();
                        [encoder setBuffer:state.environmentLineSearch offset:0 atIndex:1];
                        [encoder setBuffer:scratch offset:0 atIndex:2];
                        [encoder setBytes:&pass length:sizeof(pass) atIndex:3];
                        [encoder setBuffer:state.fgmresStates offset:0 atIndex:4];
                        [encoder setBuffer:state.statuses offset:0 atIndex:5];
                    });
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    dispatchGroups32("nm_fem_registered_release_bound_elements",(state.dispatch.tetrahedronCount+127u)/128u,[&] {
                        setDispatch();
                        [encoder setBytes:&micro length:sizeof(micro) atIndex:1];
                        [encoder setBuffer:state.objects offset:0 atIndex:2];
                        [encoder setBuffer:state.materials offset:0 atIndex:3];
                        [encoder setBuffer:state.femAccepted offset:0 atIndex:4];
                        [encoder setBuffer:state.femCandidate offset:0 atIndex:5];
                        [encoder setBuffer:state.femSolution offset:0 atIndex:6];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0 atIndex:7];
                        [encoder setBuffer:state.schedulers offset:0 atIndex:8];
                        [encoder setBuffer:state.adaptive offset:0 atIndex:9];
                        [encoder setBuffer:state.releaseFloors offset:0 atIndex:10];
                        [encoder setBuffer:state.environmentLineSearch offset:0 atIndex:11];
                        [encoder setBuffer:scratch offset:0 atIndex:12];
                        [encoder setBuffer:state.scalarPrograms offset:0 atIndex:13];
                        [encoder setBuffer:state.instructions offset:0 atIndex:14];
                        [encoder setBuffer:state.environmentParameters offset:0 atIndex:15];
                        [encoder setBytes:&budget length:sizeof(budget) atIndex:16];
                    },WorkDomain::generic,scratch,128u);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    uint32_t last=pass==3u;
                    dispatchThreads("nm_fem_registered_release_bound_finish",1u,[&] {
                        setDispatch();
                        [encoder setBuffer:state.environmentLineSearch offset:0 atIndex:1];
                        [encoder setBuffer:state.femLineSearch offset:0 atIndex:2];
                        [encoder setBuffer:scratch offset:0 atIndex:3];
                        [encoder setBuffer:state.statuses offset:0 atIndex:4];
                        [encoder setBytes:&last length:sizeof(last) atIndex:5];
                        [encoder setBytes:&budget length:sizeof(budget) atIndex:6];
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0 atIndex:7];
                    });
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    if(!last){
                        encodeHumanSupportLineSearchLimit(scratch);
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        encodeVascularLineSearchLimit(scratch);
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                }
            }
