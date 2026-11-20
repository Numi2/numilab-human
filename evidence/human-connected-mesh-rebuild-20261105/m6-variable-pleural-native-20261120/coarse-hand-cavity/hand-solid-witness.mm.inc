        if(const char* path=std::getenv("NUMI_HUMAN_HAND_CONTACT_WITNESS_CSV")) {
            require(staticBVHEnabled&&wholeBodySkinNodesPerLayer==2411u,
                    "hand solid witness requires the complete pinned paired contact banks");
            namespace witness=numi::human::initial_skin_bone_contact_witness;
            std::ofstream out(path,std::ios::out|std::ios::trunc);
            require(out.good(),"hand contact witness CSV open failed");
            out<<std::setprecision(17)
               <<"layer,row,node,body,bone,triangle,closest_distance_m,inside_rays,outside_rays,ambiguous,force_x_N,force_y_N,force_z_N\n";
            constexpr std::uint32_t flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
            for(const bool deep:{false,true}) for(const std::uint32_t row:{2261u,2262u}) {
                const auto local=(deep?wholeBodySkinBoundaryNodeMap.fatDeep:
                                        wholeBodySkinBoundaryNodeMap.outer)[row];
                const auto global=wholeBodySkinFirstGlobalFEMNode+local;
                require(global<compiled.world.fem.nodes.size(),"hand contact witness invalid compiled node");
                const auto p=witness::read3(compiled.world.fem.nodes[global].positionAndMass);
                using Triangle=NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU;
                for(std::uint32_t bankId=0u;bankId<bodyBanks.size();++bankId) {
                    const auto& bank=bodyBanks[bankId];
                    if(bank.triangles.empty()||!bodyBankEligible[bankId])continue;
                    const auto& pose=bodies[bankId];
                    const witness::Vec3 origin{pose.position.x,pose.position.y,pose.position.z};
                    const std::array<double,4> q{pose.orientation.x,pose.orientation.y,
                        pose.orientation.z,pose.orientation.w};
                    const auto localQuery=witness::rotate({-q[0],-q[1],-q[2],q[3]},
                        witness::sub(p,origin));
                    const witness::Vec3 query{float(localQuery.x),float(localQuery.y),float(localQuery.z)};
                    const std::array<double,3> comp{query.x,query.y,query.z};
                    bool close=true;
                    for(unsigned axis=0;axis<3;++axis)
                        close=close&&comp[axis]>=bank.minimum[axis]-0.015&&
                                     comp[axis]<=bank.maximum[axis]+0.015;
                    if(!close)continue;
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
                        if(d<nearest) {nearest=d;winner=&tri;bestDelta=delta;}
                    }
                    if(!winner)continue;
                    const auto distance=std::sqrt(nearest);
                    const auto parity=witness::classifyClosedSolid(query,bank.triangles);
                    const auto sample=std::find_if(fasciaBodyContactSamples.begin(),fasciaBodyContactSamples.end(),
                        [&](const auto& s){return s.slaveNode==global&&s.flags==flags;});
                    require(sample!=fasciaBodyContactSamples.end(),"hand contact witness lacks registered load sample");
                    const double area=sample->bodyLocalPointAndArea.w;
                    const double stiffness=sample->bodyLocalNormalAndReferenceSeparation.w;
                    const double pressure=parity.insideRays==3?
                        stiffness*std::max(0.0,distance-1.e-7):0.0;
                    const auto force=pressure>0.0?witness::rotate(q,
                        witness::mul(bestDelta,-area*pressure/distance)):witness::Vec3{};
                    out<<(deep?2u:0u)<<','<<row<<','<<global<<','<<bankId<<','
                       <<winner->boneStableId<<','<<winner->triangleStableId<<','<<distance<<','
                       <<parity.insideRays<<','<<parity.outsideRays<<','<<parity.ambiguous<<','
                       <<force.x<<','<<force.y<<','<<force.z<<'\n';
                }
            }
            require(out.good(),"hand contact witness CSV write failed");
        }
