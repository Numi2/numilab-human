            const char* bedWitnessOption=std::getenv("NM_FEM_REGISTERED_PLANE_LIMIT_WITNESS");
            const uint32_t bedWitnessEnabled=state.referenceAreaPlane && environments==1u &&
                bedWitnessOption && !std::strcmp(bedWitnessOption,"1") &&
                request.controlStep==0u && micro.solverIteration==15u;
            id<MTLBuffer> bedWitness=bedWitnessEnabled
                ? [state.device newBufferWithLength:32u*4u*sizeof(nm_float4) options:MTLResourceStorageModeShared]
                : state.dummy;
            if(!bedWitness){diagnostics.message="registered bed limiter observer allocation failed";return diagnostics;}
            if(bedWitnessEnabled) {
                std::memset(bedWitness.contents,0,bedWitness.length);
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> done) {
                    if(done.status!=MTLCommandBufferStatusCompleted)return;
                    const auto* rows=static_cast<const nm_float4*>(bedWitness.contents);
                    int best=-1;
                    for(int i=0;i<32;++i){
                        const auto b=rows[4*i];
                        if(b.w==1.f && (best<0 || b.x<rows[4*best].x ||
                            (b.x==rows[4*best].x && b.y<rows[4*best].y)))best=i;
                    }
                    if(best<0)return;
                    const auto* r=rows+4*best;
                    std::fprintf(stderr,"registered_plane_limit_witness={\"root\":0,\"iteration\":15,\"fraction\":%.9g,\"pair\":%u,\"node\":%u,\"gap_m\":%.9g,\"floor_m\":%.9g,\"closing_displacement_m\":%.9g,\"total_displacement_m\":%.9g,\"normal_impulse_Ns\":%.9g,\"diagonal_kg\":%.9g,\"point_xyz_m\":[%.9g,%.9g,%.9g]}\n",
                        r[0].x,unsigned(r[0].y),unsigned(r[0].z),r[1].x,r[1].y,r[1].z,r[1].w,
                        r[2].x,r[2].y,r[3].x,r[3].y,r[3].z);
                }];
            }
