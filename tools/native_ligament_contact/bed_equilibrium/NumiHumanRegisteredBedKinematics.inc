        // Section compliance remains the matched 32-section model. Only the contact
        // coordinates and their GPU Jacobians are replaced by the physical O surface.
        if (const char* setting=std::getenv("NUMI_HUMAN_LOCAL_BED_REGISTERED_OUTER")) {
            require(std::string_view(setting)=="0" || std::string_view(setting)=="1",
                    "registered outer skin option must be 0 or 1");
            if (std::string_view(setting)=="1") {
                using namespace metalrobo::human::coarse_tissue;
                require(localBedSections.size()==sourceRegionCount && useRegionalThickness &&
                        !organSupportActive && !deriveFootManifold && !ropeConfig && !bedSurface &&
                        !skin.usesRegisteredDirectWorldSurface && !femSkinSupportRequested(),
                        "registered section contacts require the sole condensed flat-bed owner");
                const auto value=[](const std::string& field) -> std::string {
                    const std::string key="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"+field;
                    const char* v=std::getenv(key.c_str());
                    require(v && *v, "missing registered contact source "+key);
                    return v;
                };
                const auto sourceHash=[](const std::filesystem::path& path) {
                    return loadedKneeSHA256Hex(loadedKneeFileSHA256(path));
                };
                const auto count=[&](const std::string& key) {
                    unsigned n=0;
                    require(registered_skin_candidate_csv_detail::number(value(key),n) && n>0,
                            "invalid registered source count "+key);
                    return n;
                };
                const auto asset=value("ROLE_O_SOURCE_ASSET");
                const auto assetHash=value("ROLE_O_SOURCE_ASSET_SHA256");
                require(sourceHash(asset)==assetHash,"outer skin provenance fingerprint mismatch");
                RegisteredSkinCurrentSourceDomain domain{value("ROLE_O_SOURCE_DOMAIN"),assetHash,
                    count("ROLE_O_SOURCE_VERTEX_COUNT"),count("ROLE_O_SOURCE_FACE_COUNT")};
                RegisteredSurface identity;
                identity.name="Registered physical outer skin";
                identity.provider="NumiHumanRegisteredSurfaceSidecarCSV";
                identity.sourceMember=asset;
                identity.sourceMeshSHA256=value("SURFACE_O_NODES_SHA256");
                identity.sourcePayloadSHA256=assetHash;
                identity.stableIdentifier=count("SURFACE_O_STABLE_IDENTIFIER");
                identity.aggregateIdentity=true;
                RegisteredSurface surface;
                RegisteredSkinCandidateCSVReport parsed;
                std::string error;
                const bool loaded=loadRegisteredSkinCandidateCSV(value("SURFACE_O_NODES_CSV"),
                    value("SURFACE_O_FACES_CSV"),restBodies.size(),identity,
                    value("SURFACE_O_FACES_SHA256"),sourceHash,surface,parsed,error,&domain);
                require(loaded,"unable to load registered physical bed surface: "+error);
                const auto n=surface.donorLocalPoints.size();
                require(n>0 && n<=UINT32_MAX && n==surface.donorBodyIndices.size(),
                        "empty or malformed registered bed surface");
                ownedSupportMaps.assign(n,MRHumanRestingVertexMap{});
                ownedSupportWeights.assign(n,MRHumanRestingInfluence{});
                regionForVertex.resize(n);
                regionCounts.assign(sourceRegionCount,0);
                for (std::uint32_t v=0;v<n;++v) {
                    const auto donor=surface.donorBodyIndices[v];
                    const auto& local=surface.donorLocalPoints[v];
                    const auto& normal=surface.outwardWorldNormals[v];
                    const auto& basis=surface.worldToDonorLocalLinear[v];
                    const mr_float4 p{float(local[0]),float(local[1]),float(local[2]),1.f};
                    const auto& body=restBodies.at(donor);
                    const auto ref=addPoint(body.position,rotatePoint(body.orientation,p));
                    float best=INFINITY; unsigned region=MR_INVALID_INDEX;
                    for (unsigned j=0;j<sourceRegionCount;++j) {
                        const auto delta=subtractPoint(ref,supportSeedCentersWorld[j]);
                        const auto d=dotPoint(delta,delta);
                        if (d<best) { best=d; region=j; }
                    }
                    require(region<sourceRegionCount,"registered O node has no source contact region");
                    ++regionCounts[region]; regionForVertex[v]=region;
                    auto& map=ownedSupportMaps[v];
                    map.deformationKind=3u; map.firstInfluence=v; map.influenceCount=1u;
                    auto& weight=ownedSupportWeights[v];
                    weight.body={donor,0u,0u,0u};
                    weight.positionAndWeight=p;
                    const auto localNormal=physical_skin_registered_surface_csv_detail::mul(basis,normal);
                    weight.normal={float(localNormal[0]),float(localNormal[1]),float(localNormal[2]),0.f};
                }
                require(std::all_of(regionCounts.begin(),regionCounts.end(),[](unsigned n){return n>0;}),
                        "registered material surface lost an anatomical bed region");
                supportMapSpan=ownedSupportMaps;
                supportWeightSpan=ownedSupportWeights;
                supportMapOffsetBytes=0u;
                supportInfluenceLayout=NumiHumanSkinInfluenceLayout::vertexMajor;
                supportMapBuffer=[device newBufferWithBytes:ownedSupportMaps.data()
                    length:supportMapSpan.size_bytes() options:MTLResourceStorageModeShared];
                supportWeightBuffer=[device newBufferWithBytes:ownedSupportWeights.data()
                    length:supportWeightSpan.size_bytes() options:MTLResourceStorageModeShared];
                require(supportMapBuffer && supportWeightBuffer,"registered material support upload failed");
                supportIdentity.text("registered-outer-source-contact-coordinates-v1");
                supportIdentity.text(identity.sourceMeshSHA256);
                supportIdentity.text(value("SURFACE_O_FACES_SHA256"));
                std::cout<<"resting_registered_bed_kinematics vertices="<<n<<" faces="<<parsed.faceRows
                         <<" regions="<<regionCounts.size()<<" outer_nodes_sha256="<<identity.sourceMeshSHA256
                         <<" contact_owner=existing_metal_stand unchanged_section_compliance=1"<<std::endl;
            }
        }
