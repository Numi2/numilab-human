                if (root==1 && environments==1u && std::getenv("NUMI_DIAG_CERTIFICATE_METALLIB")) {
                    struct Span { uint32_t offset,count,mask,id; };
                    struct Tile { float mechanics2,pressure2,maximum2; uint32_t row; };
                    std::vector<Span> spans;
                    for(uint32_t i=0;i<state.objectLayout.size();++i){
                        const auto &o=state.objectLayout[i];
                        if(o.representation==NM_REPRESENTATION_FEM && o.stateCount)
                            spans.push_back({o.stateOffset,o.stateCount,15u,i});
                    }
                    auto rigid=state.dispatch.rigidGeneralizedCapacity;
                    if(rigid)spans.push_back({2u*state.dispatch.femNodeCount+state.dispatch.mpmActiveNodeCapacity,rigid,1u,0xfffffffeu});
                    if(state.fgmresLayout.supportContactCount)
                        spans.push_back({state.fgmresLayout.supportBase,state.fgmresLayout.supportContactCount,7u,0xfffffffdu});
                    constexpr size_t tiles=128;
                    NSError* error=nil;
                    NSURL* url=[NSURL fileURLWithPath:[NSString stringWithUTF8String:std::getenv("NUMI_DIAG_CERTIFICATE_METALLIB")]];
                    id<MTLLibrary> lib=[state.device newLibraryWithURL:url error:&error];
                    id<MTLFunction> fn=[lib newFunctionWithName:@"summarize_certificate_rows"];
                    id<MTLComputePipelineState> ps=fn?[state.device newComputePipelineStateWithFunction:fn error:&error]:nil;
                    id<MTLBuffer> result=[state.device newBufferWithLength:tiles*spans.size()*sizeof(Tile) options:MTLResourceStorageModeShared];
                    if (!ps || !result || ps.threadExecutionWidth!=32 || spans.empty() || spans.size()*sizeof(Span)>4096) {
                        diagnostics.message="Cannot encode selected certificate observer"; return false;
                    }
                    for(const auto& span:spans)if((uint64_t(span.offset)+span.count)*sizeof(nm_float4)>state.femResidual.length) {
                        diagnostics.message="Certificate observer span exceeds residual buffer";return false;
                    }
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [encoder setComputePipelineState:ps];
                    [encoder setBuffer:state.femResidual offset:0 atIndex:0];
                    [encoder setBytes:spans.data() length:sizeof(Span)*spans.size() atIndex:1];
                    [encoder setBuffer:result offset:0 atIndex:2];
                    [encoder dispatchThreadgroups:MTLSizeMake(spans.size()*tiles,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb){
                        if(cb.status!=MTLCommandBufferStatusCompleted)return;
                        const auto* data=static_cast<const Tile*>(result.contents);
                        for(size_t k=0;k<spans.size();k++) {
                            double m=0,p=0;float peak=-1;uint32_t row=UINT32_MAX;
                            for(size_t j=0;j<tiles;j++) {
                                const auto &t=data[k*tiles+j];m+=t.mechanics2;p+=t.pressure2;
                                if(t.maximum2>peak || (t.maximum2==peak && t.row<row)){peak=t.maximum2;row=t.row;}
                            }
                            std::fprintf(stderr,"certificate_row_measure root=1 span=%u offset=%u count=%u mechanics2=%.12g pressure2=%.12g peak2=%.12g local=%u\n",spans[k].id,spans[k].offset,spans[k].count,m,p,double(peak),row);
                        }
                    }];
                }
