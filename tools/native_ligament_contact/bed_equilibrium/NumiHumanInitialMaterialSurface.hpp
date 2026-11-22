// Initialize the direct registered skin from the same current FEM rows used by accepted frames.
#pragma once
namespace numiInitialMaterialSurface {
template<class Object, class OuterRows, class Skin>
void bind(const Object& object, const OuterRows& rows, Skin& skin) {
    using Point = std::array<double,3>;
    require(skin.usesRegisteredDirectWorldSurface && rows.size()==skin.vertices.size() &&
            rows.size()==skin.registeredDirectPhysicalBoundaryVertexCount &&
            rows.size()==skin.registeredDirectWorldPoints.size() &&
            rows.size()==skin.registeredDirectWorldNormals.size() && skin.indices.size()%3u==0u,
            "loaded direct surface has an inconsistent node map");
    const auto area = [](const Point& a,const Point& b,const Point& c) -> Point {
        const auto x=b[0]-a[0],y=b[1]-a[1],z=b[2]-a[2];
        const auto u=c[0]-a[0],v=c[1]-a[1],w=c[2]-a[2];
        return {y*w-z*v,z*u-x*w,x*v-y*u};
    };
    std::vector<Point> positions(rows.size()),referenceAreas(rows.size()),currentAreas(rows.size());
    for (std::size_t i=0;i<rows.size();++i) {
        const auto& p=object.femNodes.at(rows[i]);
        positions[i]={p[0],p[1],p[2]};
    }
    for (std::size_t f=0;f<skin.indices.size();f+=3) {
        const auto i=skin.indices[f],j=skin.indices[f+1],k=skin.indices[f+2];
        const auto& base=skin.registeredDirectWorldPoints;
        const auto ref=area(base.at(i),base.at(j),base.at(k));
        const auto cur=area(positions.at(i),positions.at(j),positions.at(k));
        for (auto row : {i,j,k}) for(unsigned a=0;a<3;++a) {
            referenceAreas[row][a]+=ref[a];
            currentAreas[row][a]+=cur[a];
        }
    }
    for (std::size_t i=0;i<rows.size();++i) {
        auto& v=skin.vertices[i];
        v.positionX=float(positions[i][0]);v.positionY=float(positions[i][1]);v.positionZ=float(positions[i][2]);
        // Apply the shortest rotation of the geometric normal to the authored normal.
        // A rigid translation therefore retains its original shading.
        auto r=referenceAreas[i],c=currentAreas[i];
        const double nr=std::hypot(r[0],r[1],r[2]),nc=std::hypot(c[0],c[1],c[2]);
        require(nr>0 && nc>0,"registered skin has an undefined area normal");
        for(unsigned a=0;a<3;++a){r[a]/=nr;c[a]/=nc;}
        const Point origin{};
        const auto axis=area(origin,r,c);
        const auto n=skin.registeredDirectWorldNormals[i];
        const auto first=area(origin,axis,n),second=area(origin,axis,first);
        const double d=1.+r[0]*c[0]+r[1]*c[1]+r[2]*c[2];
        require(d>1.e-8,"registered skin has a reversed normal neighborhood");
        Point updated{};
        for(unsigned a=0;a<3;++a)updated[a]=n[a]+first[a]+second[a]/d;
        const double length=std::hypot(updated[0],updated[1],updated[2]);
        require(length>0 && std::isfinite(length),"invalid loaded surface normal");
        for(double& a:updated)a/=length;
        skin.registeredDirectWorldNormals[i]=updated;
        v.normalX=float(updated[0]);v.normalY=float(updated[1]);v.normalZ=float(updated[2]);
    }
    skin.registeredDirectWorldPoints=std::move(positions);
    std::cout<<"registered_initial_display=physical_outer_volume rows="<<rows.size()<<std::endl;
}
}
