                    if(candidate->releaseVolumeMode){
                        std::vector<nm_float4> floors;
                        floors.reserve(world.fem.tetrahedra.size());
                        for(const auto& tet:world.fem.tetrahedra){
                            const auto id=tet.nodes;const std::uint32_t ids[4]={id.x,id.y,id.z,id.w};
                            bool reference=true;
                            for(auto i:ids) reference&=world.fem.nodes.at(i).referenceDisplacementAndMode.w==1.f;
                            const auto& a=world.fem.nodes[ids[0]];
                            auto determinant=[&](bool loaded){
                                double e[3][3]{};
                                for(unsigned j=1;j<4;++j){
                                    const auto& b=world.fem.nodes[ids[j]];
                                    auto diff=[&](float nm_float4::*v){
                                        if(!loaded)return double(b.restAndFixed.*v)-double(a.restAndFixed.*v);
                                        if(!reference)return double(b.positionAndMass.*v)-double(a.positionAndMass.*v);
                                        return (double(b.restAndFixed.*v)-double(a.restAndFixed.*v))+
                                            (double(b.referenceDisplacementAndMode.*v)-double(a.referenceDisplacementAndMode.*v))+
                                            (double(b.referenceDisplacementLow.*v)-double(a.referenceDisplacementLow.*v));
                                    };
                                    e[j-1][0]=diff(&nm_float4::x);e[j-1][1]=diff(&nm_float4::y);e[j-1][2]=diff(&nm_float4::z);
                                }
                                return e[0][0]*(e[1][1]*e[2][2]-e[1][2]*e[2][1])-
                                       e[0][1]*(e[1][0]*e[2][2]-e[1][2]*e[2][0])+
                                       e[0][2]*(e[1][0]*e[2][1]-e[1][1]*e[2][0]);
                            };
                            const double start=determinant(true),rest=determinant(false);
                            if(!(tet.identity.w & NM_OBJECT_ACTIVE)){floors.push_back({});continue;}
                            if(!std::isfinite(start)||!std::isfinite(rest)||start*rest<=0.){
                                diagnostics.message="invalid registered release tetrahedron";return diagnostics;
                            }
                            const double floor=.35*start;
                            const float high=float(floor);
                            const float materialFloor=candidate->releaseVolumeMode==2u?
                                std::nextafter(float(floor/rest),std::numeric_limits<float>::infinity()):0.f;
                            floors.push_back({high,float(floor-double(high)),materialFloor,start>0.?1.f:-1.f});
                        }
                        candidate->releaseFloors=[candidate->device newBufferWithBytes:floors.data() length:floors.size()*sizeof(nm_float4) options:MTLResourceStorageModeShared];
                        candidate->releaseFailingRow=privateScratch<std::uint32_t>(candidate->device,1u,valid,candidate->residentBytes);
                        if(!candidate->releaseFloors || !candidate->releaseFailingRow){diagnostics.message="failed to allocate registered release bounds";return diagnostics;}
                        candidate->residentBytes+=candidate->releaseFloors.allocatedSize;
                    }
