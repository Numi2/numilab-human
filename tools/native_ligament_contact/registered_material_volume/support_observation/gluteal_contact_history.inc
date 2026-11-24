if (const char* request=std::getenv("NUMI_HUMAN_GLUTEAL_FAT_TISSUE_AUDIT"); request && *request) {
    const char* prefix=std::getenv("NUMI_HUMAN_REGISTERED_CELL_NODE_PREFIX");
    require(prefix && *prefix && continuumTransaction->posteriorBedNodes.size()==a.contactHistories.size(),
            "Accepted bed history lacks its compiled node map");
    std::ofstream active(std::string(prefix)+"-posterior-bed-history.csv");
    require(active.good(),"Unable to open accepted bed history");
    active<<"contact_row,registered_skin_node,terminal_microstep_normal_impulse_Ns\n"<<std::setprecision(17);
    for (size_t i=0;i<a.contactHistories.size();++i) {
        const auto normalImpulse=a.contactHistories[i].w;
        if (normalImpulse>0.f) active<<i<<','<<continuumTransaction->posteriorBedNodes[i]<<','<<normalImpulse<<'\n';
    }
    require(active.good(),"Could not write accepted bed history");
}
