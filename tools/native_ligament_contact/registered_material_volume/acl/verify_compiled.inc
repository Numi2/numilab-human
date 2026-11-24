    if (const auto setting=std::getenv("NUMI_HUMAN_ACL_REFERENCE_DISPLACEMENT");
        setting && std::string_view(setting)=="1") {
        require(aclObjects.size()==2,"missing reference ACL pair");
        for (std::size_t side=0;side<aclObjects.size();++side) {
            const auto& object=aclObjects[side].object;
            const auto& descriptor=compiled.world.objects.at(side);
            const auto& authored=source.materials.at(object.materialIndex);
            const auto& material=compiled.world.materials.at(object.materialIndex);
            require(descriptor.stateOffset==16*side && descriptor.stateCount==16 &&
                    descriptor.elementCount==18 && (descriptor.flags & NM_OBJECT_FEM_REFERENCE_DISPLACEMENT_GRADIENT),
                    "ACL compiled displacement owner differs from the source");
            for (std::size_t i=0;i<authored.parameters.size();++i)
                require(compiled.world.parameters.at(material.parameterOffset+i).valueAndBounds.x==
                        float(authored.parameters[i].defaultValue),"ACL material changed in compilation");
            for (unsigned i=0;i<descriptor.stateCount;++i) {
                const auto& node=compiled.world.fem.nodes.at(descriptor.stateOffset+i);
                require(node.referenceDisplacementAndMode.w==1.0f &&
                        node.referenceDisplacementAndMode.x==0.0f &&
                        node.referenceDisplacementAndMode.y==0.0f &&
                        node.referenceDisplacementAndMode.z==0.0f,
                        "ACL compiled reference differs from its anatomical initial condition");
            }
            std::cout<<"resting_acl_reference_gradient=compiled side="<<side
                     <<" state_offset="<<descriptor.stateOffset<<" material_parameters="
                     <<authored.parameters.size()<<" initial_displacement_zero=1"<<std::endl;
        }
    }
