            // Snapshot the actual assembled impulse row and its kinematic owner together.
            // This branch runs only for an explicitly selected forensic node.
            if (const char* rawNode=std::getenv("NM_FEM_INITIAL_CONTACT_ROW"); rawNode &&
                state.referenceAreaPlane && request.controlStep==0u && micro.solverIteration==15u) {
                uint32_t nodeIndex=0;
                const auto end=rawNode+std::strlen(rawNode);
                const auto parsed=std::from_chars(rawNode,end,nodeIndex);
                if(parsed.ec!=std::errc{} || parsed.ptr!=end || nodeIndex>=state.dispatch.femNodeCount ||
                    environments!=1u) { diagnostics.message="invalid forensic FEM node";return diagnostics; }
                const NSUInteger bytes=sizeof(NMFEMNodeStateGPU)+sizeof(nm_float4);
                id<MTLBuffer> snapshot=[state.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if(snapshot==nil) {diagnostics.message="failed FEM normal-row snapshot allocation";return diagnostics;}
                [encoder endEncoding];encoder=nil;
                id<MTLBlitCommandEncoder> copy=[commandBuffer blitCommandEncoder];
                [copy copyFromBuffer:state.femCandidate sourceOffset:sizeof(NMFEMNodeStateGPU)*nodeIndex
                    toBuffer:snapshot destinationOffset:0 size:sizeof(NMFEMNodeStateGPU)];
                [copy copyFromBuffer:state.femResidual sourceOffset:sizeof(nm_float4)*nodeIndex
                    toBuffer:snapshot destinationOffset:sizeof(NMFEMNodeStateGPU) size:sizeof(nm_float4)];
                [copy endEncoding];
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> done){
                    if(done.status!=MTLCommandBufferStatusCompleted)return;
                    auto* n=static_cast<const NMFEMNodeStateGPU*>(snapshot.contents);
                    const auto r=*reinterpret_cast<const nm_float4*>(n+1);
                    std::fprintf(stderr,"registered_contact_raw_row={\"node\":%u,\"root\":0,\"newton\":15,\"raw_rhs_Ns\":[%.9g,%.9g,%.9g],\"mass_kg\":%.9g,\"point_m\":[%.9g,%.9g,%.9g],\"velocity_m_s\":[%.9g,%.9g,%.9g]}\n",
                        nodeIndex,r.x,r.y,r.z,n->positionAndMass.w,n->positionAndMass.x,n->positionAndMass.y,
                        n->positionAndMass.z,n->velocityAndInverseMass.x,n->velocityAndInverseMass.y,n->velocityAndInverseMass.z);
                }];
                encoder=[commandBuffer computeCommandEncoder];
                if(encoder==nil){diagnostics.message="failed to resume after forensic FEM row";return diagnostics;}
            }
