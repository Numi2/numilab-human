if (const char* request=std::getenv("NUMI_HUMAN_GLUTEAL_FAT_TISSUE_AUDIT"); request && *request) {
    auto& map=tissueOwner.transaction.posteriorBedNodes;
    const auto& outer=tissueOwner.wholeBodySkinBoundaryNodeMap.outer;
    const auto& rows=finalWorld.contact.pairs;
    require(map.empty() && rows.size()==outer.size() && finalWorld.contact.rigidProxies.size()==1u &&
            finalWorld.contact.rigidProxies.front().shapeKind==NM_RIGID_PLANE,
            "Posterior history requires the registered bed and full exterior");
    for (const auto& p:rows) {
        const auto base=finalWorld.mpm.nodes.size()+tissueOwner.wholeBodySkinFirstGlobalFEMNode;
        require(p.continuumNode>=base && p.rigidProxy==0,"Bed history row has an unrelated owner");
        map.push_back(uint32_t(p.continuumNode-base));
    }
    auto expected=outer;
    auto actual=map;
    std::sort(expected.begin(),expected.end()); std::sort(actual.begin(),actual.end());
    require(expected.size()==actual.size() && std::equal(expected.begin(),expected.end(),actual.begin()),
            "Bed history does not cover exactly the registered outer skin");
}
