        const char* handClearanceSetting=std::getenv("NUMI_HUMAN_HAND_CLEARANCE_TRIAL");
        if(handClearanceSetting) {
            struct ChangedPoint {std::size_t role,row; V3 reference, shift;};
            constexpr std::array<ChangedPoint,POINT_COUNT> fit{{
POINTS
            }};
            require(std::string_view(handClearanceSetting)=="1"&&enableWholeBodySkinFat&&
                    !registeredVolumeInstallRequested&&region.worldPoints.size()==2411u,
                    "hand clearance trial requires the exact diagnostic coarse parent");
            const std::array<std::vector<V3>*,3> world{&region.worldPoints,
                &skinLayer.innerWorld,&fatLayer.innerWorld};
            const std::array<std::vector<V3>*,3> local{&region.donorLocalPoints,
                &skinLayer.innerLocal,&fatLayer.innerLocal};
            std::vector<std::uint8_t> touched(region.worldPoints.size(),0u);
            std::vector<std::uint8_t> outerTouched(region.worldPoints.size(),0u);
            for(const auto& point:fit)
                require(norm(sub((*world[point.role])[point.row],point.reference))<=2.0e-10,
                    "hand clearance trial node row differs from compiled parent source");
            for(const auto& point:fit) {
                const auto row=point.row;
                const auto& frame=region.worldToDonorLocalLinear[row];
                V3 localShift{};
                for(unsigned i=0;i<3u;++i)
                    for(unsigned j=0;j<3u;++j)
                        localShift[i]+=frame[3u*i+j]*point.shift[j];
                auto& p=(*world[point.role])[row];
                p=add(p,point.shift);
                auto& x=(*local[point.role])[row];
                x=add(x,localShift);
                touched[row]=1u;
                if(point.role==0u&&norm(point.shift)>0.0)outerTouched[row]=1u;
            }
            skinLayer.outerWorld=region.worldPoints;
            skinLayer.outerLocal=region.donorLocalPoints;
            fatLayer.outerWorld=skinLayer.innerWorld;
            fatLayer.outerLocal=skinLayer.innerLocal;
            std::vector<std::uint8_t> normalTouched=outerTouched;
            for(const auto& face:region.faces)
                if(outerTouched[face[0]]||outerTouched[face[1]]||outerTouched[face[2]])
                    for(const auto row:face)normalTouched[row]=1u;
            std::vector<V3> normals(region.worldPoints.size());
            for(const auto& face:region.faces) {
                if(!normalTouched[face[0]]&&!normalTouched[face[1]]&&!normalTouched[face[2]])continue;
                const auto p=region.worldPoints[face[0]];
                const auto normal=cross(sub(region.worldPoints[face[1]],p),
                                        sub(region.worldPoints[face[2]],p));
                for(const auto row:face)normals[row]=add(normals[row],normal);
            }
            for(std::size_t row=0;row<touched.size();++row) {
                if(normalTouched[row]) {
                    const double length=norm(normals[row]);
                    require(std::isfinite(length)&&length>0.0,"hand clearance invalid outer normal");
                    region.outwardWorldNormals[row]=mul(normals[row],1.0/length);
                }
                if(!touched[row])continue;
                skinThickness[row]=norm(sub(skinLayer.outerWorld[row],skinLayer.innerWorld[row]));
                fatThickness[row]=norm(sub(fatLayer.outerWorld[row],fatLayer.innerWorld[row]));
                skinLayer.thicknessByVertexM[row]=skinThickness[row];
                fatLayer.thicknessByVertexM[row]=fatThickness[row];
            }
            const auto s=std::minmax_element(skinThickness.begin(),skinThickness.end());
            const auto f=std::minmax_element(fatThickness.begin(),fatThickness.end());
            skinLayer.minThicknessM=*s.first;skinLayer.maxThicknessM=*s.second;
            fatLayer.minThicknessM=*f.first;fatLayer.maxThicknessM=*f.second;
            std::cout<<"coarse_hand_clearance_trial moved_nodes="<<fit.size()<<" fit_sha256=FIT_SHA\n";
        }
