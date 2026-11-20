        if(const char* path=std::getenv("NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV")) {
            require(staticBVHEnabled&&wholeBodySkinNodesPerLayer==2411u,
                    "batch solid audit requires the pinned paired contact banks");
            namespace witness=numi::human::initial_skin_bone_contact_witness;
            using Triangle=NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU;
            std::ofstream out(path,std::ios::out|std::ios::trunc);
            require(out.good(),"batch contact witness CSV open failed");
            out<<std::setprecision(17)
               <<"layer,row,node,body,bone,triangle,closest_distance_m,inside_rays,outside_rays,"
                 "ambiguous,has_contact_sample,normal_x,normal_y,normal_z,force_N\n";
            constexpr std::uint32_t flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
            std::vector<const NMNumiHumanFEMBodyContactSampleGPU*> samples(compiled.world.fem.nodes.size(),nullptr);
            for(const auto& sample:fasciaBodyContactSamples)
                if(sample.flags==flags&&sample.slaveNode<samples.size())samples[sample.slaveNode]=&sample;
            std::size_t tested=0,inside=0,ambiguous=0,eligibleBanks=0;
            for(std::uint32_t bankId=0u;bankId<bodyBanks.size();++bankId) {
                const auto& bank=bodyBanks[bankId];
                if(bank.triangles.empty()||!bodyBankEligible[bankId])continue;
                ++eligibleBanks;
                const auto& pose=bodies[bankId];
                const witness::Vec3 origin{pose.position.x,pose.position.y,pose.position.z};
                const std::array<double,4> q{pose.orientation.x,pose.orientation.y,
                    pose.orientation.z,pose.orientation.w};
                const std::array<double,4> inv{-q[0],-q[1],-q[2],q[3]};
                for(const bool deep:{false,true}) for(std::uint32_t row=0;row<wholeBodySkinNodesPerLayer;++row) {
                    const auto local=(deep?wholeBodySkinBoundaryNodeMap.fatDeep:
                                            wholeBodySkinBoundaryNodeMap.outer)[row];
                    const auto global=wholeBodySkinFirstGlobalFEMNode+local;
                    require(global<samples.size(),"batch contact witness invalid compiled node");
                    const auto p=witness::read3(compiled.world.fem.nodes[global].positionAndMass);
                    const auto relative=witness::rotate(inv,witness::sub(p,origin));
                    const witness::Vec3 query{float(relative.x),float(relative.y),float(relative.z)};
                    const std::array<double,3> xyz{query.x,query.y,query.z};
                    bool candidate=true;
                    for(unsigned axis=0;axis<3;++axis)
                        candidate=candidate&&xyz[axis]>=bank.minimum[axis]&&xyz[axis]<=bank.maximum[axis];
                    if(!candidate)continue;
                    ++tested;
                    const auto parity=witness::classifyClosedSolid(query,bank.triangles);
                    if(parity.outsideRays==3&&!parity.ambiguous)continue;
                    inside+=parity.insideRays==3;
                    ambiguous+=parity.ambiguous||parity.insideRays!=3;
                    double nearest=std::numeric_limits<double>::infinity();
                    const Triangle* winner=nullptr;
                    witness::Vec3 bestDelta{};
                    for(const auto& tri:bank.triangles) {
                        const auto c=witness::closestPointOnTriangle(query,
                            witness::read3(tri.bodyLocalVertex0),witness::read3(tri.bodyLocalVertex1),
                            witness::read3(tri.bodyLocalVertex2));
                        if(!c.valid)continue;
                        const auto delta=witness::sub(query,c.point);
                        const auto d=witness::norm2(delta);
                        if(d<nearest||(d==nearest&&winner&&
                            std::pair{tri.boneStableId,tri.triangleStableId}<
                            std::pair{winner->boneStableId,winner->triangleStableId})) {
                            nearest=d;winner=&tri;bestDelta=delta;
                        }
                    }
                    require(winner!=nullptr,"batch contact witness found no valid triangle");
                    const double distance=std::sqrt(nearest);
                    const auto normal=parity.insideRays==3&&distance>0.0?
                        witness::rotate(q,witness::mul(bestDelta,-1.0/distance)):witness::Vec3{};
                    const auto* sample=samples[global];
                    const double force=parity.insideRays==3&&sample?
                        sample->bodyLocalPointAndArea.w*sample->bodyLocalNormalAndReferenceSeparation.w*
                        std::max(0.0,distance-1.e-7):0.0;
                    out<<(deep?2u:0u)<<','<<row<<','<<global<<','<<bankId<<','
                       <<winner->boneStableId<<','<<winner->triangleStableId<<','<<distance<<','
                       <<parity.insideRays<<','<<parity.outsideRays<<','<<parity.ambiguous<<','
                       <<(sample!=nullptr)<<','<<normal.x<<','<<normal.y<<','<<normal.z<<','<<force<<'\n';
                }
            }
            out.flush();
            require(out.good(),"batch contact witness CSV write failed");
            std::cout<<"initial_skin_bone_solid_audit"
                     <<" eligible_banks="<<eligibleBanks<<" roles=outer,fat_deep"
                     <<" total_points="<<2u*wholeBodySkinNodesPerLayer
                     <<" aabb_point_bank_tests="<<tested<<" inside_pairs="<<inside
                     <<" ambiguous_pairs="<<ambiguous<<std::endl;
        }
