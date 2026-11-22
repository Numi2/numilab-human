// Build a conservative reference-area measure directly from the compiled FEM boundary.
struct PlaneSurfaceInventory {
    std::vector<std::array<uint32_t, 4>> faces;
    std::vector<std::array<float, 4>> measure;
    std::vector<std::array<uint32_t, 2>> range;
    std::vector<uint32_t> corners;
};

template<class World>
static PlaneSurfaceInventory makePlaneSurfaceInventory(const World& world) {
    const auto& d=world.dispatch;
    const auto& nodes=world.fem.nodes;
    const auto& pairs=world.contact.pairs;
    auto demand=[](bool ok,const char* msg){if(!ok) throw std::runtime_error(msg);};
    demand(d.environmentCount==1 && world.contact.rigidProxies.size()==1 && !pairs.empty(),
           "registered plane requires one human and its unique fixed plane");
    const auto& plane=world.contact.rigidProxies.front();
    demand(plane.shapeKind==NM_RIGID_PLANE && !plane.flags && plane.localCenterAndRadius.x==0.f &&
           plane.localCenterAndRadius.y==0.f && plane.localCenterAndRadius.z==1.f,
           "registered plane requires an upward fixed plane");
    const auto object=pairs.front().objectIndex;
    std::vector<uint32_t> owner(nodes.size(),NM_INVALID_INDEX);
    for(uint32_t i=0;i<pairs.size();++i){
        const auto& p=pairs[i];
        demand(p.objectIndex==object && p.rigidProxy==0 && p.continuumNode>=d.gridNodeCount,
               "heterogeneous plane pairs");
        const auto n=p.continuumNode-d.gridNodeCount;
        demand(n<nodes.size() && owner[n]==NM_INVALID_INDEX && nodes[n].positionAndMass.w>0,
               "invalid or multiply-owned bed node");
        owner[n]=i;
    }
    PlaneSurfaceInventory out;
    std::vector<double> dual(nodes.size(),0.);
    out.range.resize(pairs.size(),{0,0});
    for(const auto& f:world.fem.surfaceFaces){
        if(f.adjacency.z!=object || f.adjacency.y!=NM_INVALID_INDEX) continue;
        const auto& ids=world.fem.tetrahedra.at(f.adjacency.x).nodes;
        const uint32_t tet[4]={ids.x,ids.y,ids.z,ids.w};
        demand(f.sides.x<4,"invalid exterior face index");
        std::array<uint32_t,4> tri{};
        unsigned j=0;
        for(unsigned i=0;i<4;++i) if(i!=f.sides.x) tri[j++]=tet[i];
        if(owner.at(tri[0])==NM_INVALID_INDEX || owner.at(tri[1])==NM_INVALID_INDEX ||
           owner.at(tri[2])==NM_INVALID_INDEX) continue;
        const auto& a=nodes[tri[0]].restAndFixed;
        const auto& b=nodes[tri[1]].restAndFixed;
        const auto& c=nodes[tri[2]].restAndFixed;
        const double u[3]={double(b.x)-a.x,double(b.y)-a.y,double(b.z)-a.z};
        const double v[3]={double(c.x)-a.x,double(c.y)-a.y,double(c.z)-a.z};
        const double ar=.5*std::hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);
        demand(std::isfinite(ar) && ar>0,"degenerate registered bed face");
        out.faces.push_back(tri);out.measure.push_back({float(ar),0,0,0});
        for(unsigned k=0;k<3;++k){dual[tri[k]]+=ar/3.;++out.range[owner[tri[k]]][1];}
    }
    demand(!out.faces.empty() && out.faces.size()<UINT32_MAX/3,"invalid surface extent");
    std::vector<uint32_t> next;
    uint32_t offset=0;
    for(auto& r:out.range){demand(r[1]>0,"contact node missing a material face");r[0]=offset;next.push_back(offset);offset+=r[1];}
    out.corners.resize(offset);
    for(uint32_t f=0;f<out.faces.size();++f) for(uint32_t k=0;k<3;++k){
        const uint32_t n=out.faces[f][k];
        out.measure[f][1+k]=float(1./dual[n]);
        demand(std::isfinite(out.measure[f][1+k]),"invalid surface dual area");
        out.corners[next[owner[n]]++]=3*f+k;
    }
    return out;
}
