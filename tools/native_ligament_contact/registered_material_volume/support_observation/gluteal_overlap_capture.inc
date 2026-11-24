if (const char* request = std::getenv("NUMI_HUMAN_GLUTEAL_FAT_TISSUE_AUDIT"); request && *request) {
    std::ifstream limits(request);
    std::array<double,3> lo{}, hi{};
    for (auto& v:lo) require(bool(limits >> v) && std::isfinite(v),"Invalid posterior audit lower bound");
    for (auto& v:hi) require(bool(limits >> v) && std::isfinite(v),"Invalid posterior audit upper bound");
    for (unsigned i=0;i<3;++i) require(lo[i] < hi[i],"Empty posterior audit box");
    const char* prefix=std::getenv("NUMI_HUMAN_REGISTERED_CELL_NODE_PREFIX");
    require(prefix && *prefix && measureShape,"Posterior audit requires the accepted skin/fat capture");
    std::ofstream cells(std::string(prefix)+"-posterior-tissue-cells.csv");
    std::ofstream objects(std::string(prefix)+"-posterior-tissue-objects.csv");
    require(bool(cells) && bool(objects),"Unable to open posterior material audit");
    cells << "object,global_cell,x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3\n" << std::setprecision(17);
    objects << "object,name,total_cells,overlapping_bounds,skin_fat\n";
    size_t totalCells=0;
    const auto& ranges=continuumTransaction->restingFEMAuditRanges;
    for (size_t object=0;object<ranges.size();++object) {
        const auto& o=ranges[object];
        const bool registered=o.firstTetrahedron==firstCell && o.tetrahedronCount==regionCells;
        size_t overlap=0;
        require(o.firstTetrahedron==totalCells && totalCells+o.tetrahedronCount<=a.femTopologyTetrahedra.size(),
                "Discontinuous material audit ownership");
        totalCells+=o.tetrahedronCount;
        if (!registered) for (size_t ordinal=o.firstTetrahedron;ordinal<totalCells;++ordinal) {
            const auto& t=a.femTopologyTetrahedra[ordinal].nodes;
            const std::array<unsigned,4> ids{t.x,t.y,t.z,t.w};
            std::array<std::array<double,3>,4> xyz{};
            for (unsigned j=0;j<4;++j) {
                const auto& n=a.femNodes.at(ids[j]);
                const auto& p=n.positionAndMass;
                const auto& r=n.restAndFixed;
                const auto& u=n.referenceDisplacementAndMode;
                const auto& l=n.referenceDisplacementLow;
                xyz[j]=u.w==1.f ? std::array<double,3>{(double(r.x)+u.x)+l.x,(double(r.y)+u.y)+l.y,(double(r.z)+u.z)+l.z}
                                 : std::array<double,3>{p.x,p.y,p.z};
            }
            bool hit=true;
            for (unsigned k=0;k<3;++k) {
                auto lower=xyz[0][k], upper=lower;
                for (const auto& point:xyz) {
                    require(std::isfinite(point[k]),"Nonfinite accepted posterior audit point");
                    lower=std::min(lower,point[k]); upper=std::max(upper,point[k]);
                }
                hit=hit && lower<=hi[k] && upper>=lo[k];
            }
            if (hit) {
                cells << object << ',' << ordinal;
                for (const auto& point:xyz) for (double v:point) cells << ',' << v;
                cells << '\n';
                ++overlap;
            }
        }
        objects << object << ',' << std::quoted(o.objectName) << ',' << o.tetrahedronCount
                << ',' << overlap << ',' << registered << '\n';
    }
    require(totalCells==a.femTopologyTetrahedra.size() && cells.good() && objects.good(),
            "Incomplete accepted posterior audit");
}
