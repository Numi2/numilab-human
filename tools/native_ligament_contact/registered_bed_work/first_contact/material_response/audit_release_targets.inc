                        if (coordinates.data()==poseQ.data() &&
                            std::getenv("NUMI_HUMAN_FIRST_ROOT_RELEASE_TARGETS")!=nullptr) {
                            const auto& owner=*restingPassiveFasciaOwner;
                            const auto& nodes=owner.transaction.initial.femNodes;
                            std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> anchors;
                            std::vector<size_t> ids;
                            require(nodes.size()==owner.nodeAnchors.size(),"Incomplete attachment map");
                            for(size_t n=0;n<nodes.size();++n) {
                                const auto& anchor=owner.nodeAnchors[n];
                                if(anchor.flags!=NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) continue;
                                require(anchor.bodyIndex<result.bodyPoses.size(),"Invalid attachment body");
                                anchors.push_back(anchor);
                                ids.push_back(n);
                            }
                            require(!ids.empty(),"No registered fixed attachments");
                            const char* path=std::getenv("NUMI_MATTER_METALLIB_OVERRIDE");
                            require(path && *path,"Release target audit requires the actual pinned metallib");
                            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
                            NSError* err=nil;
                            id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:
                                [NSString stringWithUTF8String:path]] error:&err];
                            id<MTLFunction> fn=[lib newFunctionWithName:@"numi_matter_metal::nm_numi_human_assemble_fem_kinematic_targets"];
                            id<MTLComputePipelineState> pso=fn?[device newComputePipelineStateWithFunction:fn error:&err]:nil;
                            require(pso!=nil,"Cannot load the coupled FEM anchor projection kernel");
                            const auto options=MTLResourceStorageModeShared;
                            id<MTLBuffer> src=[device newBufferWithBytes:anchors.data()
                                length:anchors.size()*sizeof(anchors[0]) options:options];
                            id<MTLBuffer> bodies=[device newBufferWithBytes:result.bodyPoses.data()
                                length:result.bodyPoses.size()*sizeof(result.bodyPoses[0]) options:options];
                            id<MTLBuffer> out=[device newBufferWithLength:ids.size()*sizeof(nm_float4) options:options];
                            require(src && bodies && out,"Release target audit buffers unavailable");
                            NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
                            dispatch.environmentCount=1;
                            dispatch.femNodeCount=static_cast<uint32_t>(ids.size());
                            dispatch.bodyPoseStride=static_cast<uint32_t>(result.bodyPoses.size());
                            id<MTLCommandQueue> queue=[device newCommandQueue];
                            id<MTLCommandBuffer> command=[queue commandBuffer];
                            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
                            [encoder setComputePipelineState:pso];
                            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
                            [encoder setBuffer:src offset:0 atIndex:1];
                            [encoder setBuffer:bodies offset:0 atIndex:2];
                            [encoder setBuffer:out offset:0 atIndex:3];
                            [encoder dispatchThreads:MTLSizeMake(ids.size(),1,1)
                                threadsPerThreadgroup:MTLSizeMake(pso.threadExecutionWidth,1,1)];
                            [encoder endEncoding];[command commit];[command waitUntilCompleted];
                            require(command.status==MTLCommandBufferStatusCompleted,"Release projection failed");
                            const auto* target=static_cast<const nm_float4*>(out.contents);
                            const auto flags=std::cout.flags();const auto precision=std::cout.precision();
                            for(size_t i=0;i<ids.size();++i) {
                                const auto n=ids[i];
                                const auto& s=nodes[n];
                                const auto h=s.referenceDisplacementAndMode,l=s.referenceDisplacementLow;
                                const auto p=s.positionAndMass, rest=s.restAndFixed, v=s.velocityAndInverseMass;
                                require(target[i].w==1,"Release target inactive");
                                const double x=h.w==1?double(rest.x)+double(h.x)+l.x:p.x;
                                const double y=h.w==1?double(rest.y)+double(h.y)+l.y:p.y;
                                const double z=h.w==1?double(rest.z)+double(h.z)+l.z:p.z;
                                std::cout<<std::setprecision(17)<<"first_root_release_target="<<n<<" "<<anchors[i].bodyIndex
                                    <<" "<<p.w<<" "<<double(target[i].x)-x<<" "<<double(target[i].y)-y
                                    <<" "<<double(target[i].z)-z<<" "<<v.x<<" "<<v.y<<" "<<v.z<<"\n";
                            }
                            std::cout.flags(flags);std::cout.precision(precision);
                        }
