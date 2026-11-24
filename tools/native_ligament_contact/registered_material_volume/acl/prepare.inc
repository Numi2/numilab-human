            const auto aclReference=std::getenv("NUMI_HUMAN_ACL_REFERENCE_DISPLACEMENT");
            if (aclReference && std::string_view(aclReference)=="1") {
                auto& object=built.object;
                require(object.femReferenceNodes.empty() && object.femNodes.size()==16 &&
                        object.tetrahedra.size()==18 && !object.mixedFEM && !object.multiphysics.enabled,
                        "registered ACL reference requires the existing bilateral continuum model");
                object.femReferenceNodes=object.femNodes;
                std::vector<std::byte> identity;
                const auto append=[&](auto slice) {
                    const auto b=std::as_bytes(slice);
                    identity.insert(identity.end(),b.begin(),b.end());
                };
                append(std::span(bones->payloadSha256));
                append(std::span(&side,1));
                append(std::span(object.femReferenceNodes));
                append(std::span(object.tetrahedra));
                const auto digest=loadedKneeSHA256(identity.data(),identity.size());
                for (unsigned word=0;word<4;++word) {
                    std::uint64_t value=0;
                    for (unsigned byte=0;byte<8;++byte) value=(value<<8)|digest[8*word+byte];
                    object.femReferenceSourceIdentity[word]=value;
                }
                require(!object.identifiable,"the reference ACL uses the fixed-parameter resting scene");
                // Preserve the source parameter values and intervals. This attached runtime does not run parameter identification.
                for (auto& parameter:built.material.parameters) parameter.identifiable=false;
                object.femMaterialIndices.assign(object.tetrahedra.size(),object.materialIndex);
                identity.clear();
                const auto materialFile=loadedKneeFileSHA256(NUMI_HUMAN_OPEN_KNEE_LIGAMENT_MATERIAL);
                append(std::span(materialFile));
                append(std::span(digest));
                append(std::span(object.femMaterialIndices));
                for (const auto& parameter:built.material.parameters) {
                    const std::uint64_t length=parameter.name.size();
                    append(std::span(&length,1));
                    append(std::span(parameter.name));
                    append(std::span(&parameter.dimension,1));
                    append(std::span(&parameter.defaultValue,1));
                    append(std::span(&parameter.lower,1));
                    append(std::span(&parameter.upper,1));
                    append(std::span(&parameter.logarithmic,1));
                    append(std::span(&parameter.identifiable,1));
                }
                const auto materialIdentity=loadedKneeSHA256(identity.data(),identity.size());
                for (unsigned word=0;word<4;++word) {
                    std::uint64_t value=0;
                    for (unsigned byte=0;byte<8;++byte) value=(value<<8)|materialIdentity[8*word+byte];
                    object.femMaterialSourceIdentity[word]=value;
                }
                require(object.femHumanAttachments.empty(),"unexpected second ACL attachment owner");
                object.femReferenceDisplacementGradient=true;
                std::cout<<"resting_acl_reference_gradient=bound side="<<side
                         <<" source="<<loadedKneeSHA256Hex(digest)<<" nodes="<<object.femNodes.size()
                         <<" tetrahedra="<<object.tetrahedra.size()<<std::endl;
            }
