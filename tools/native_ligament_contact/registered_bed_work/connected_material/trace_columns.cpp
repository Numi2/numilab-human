// Traverse native tetrahedral adjacency along a vertical material line at authoring time.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
using Key = std::uint64_t;
struct Face { Key key; int slot; };
Key key(int a, int b, int c) {
    if(a>b) std::swap(a,b);
    if(b>c) std::swap(b,c);
    if(a>b) std::swap(a,b);
    return (Key(a)<<40) | (Key(b)<<20) | Key(c);
}
std::array<int,3> face(const int* cell,int opposite) {
    return {cell[(opposite+1)%4],cell[(opposite+2)%4],cell[(opposite+3)%4]};
}
double det(const double* x,const int* cell) {
    const double* a=x+3*cell[0]; const double* b=x+3*cell[1];
    const double* c=x+3*cell[2]; const double* d=x+3*cell[3];
    const double ux=b[0]-a[0],uy=b[1]-a[1],uz=b[2]-a[2];
    const double vx=c[0]-a[0],vy=c[1]-a[1],vz=c[2]-a[2];
    const double wx=d[0]-a[0],wy=d[1]-a[1],wz=d[2]-a[2];
    return ux*(vy*wz-vz*wy)+uy*(vz*wx-vx*wz)+uz*(vx*wy-vy*wx);
}
bool height(const double* x,const std::array<int,3>& f,double px,double py,double& z) {
    const double* a=x+3*f[0];const double* b=x+3*f[1]; const double* c=x+3*f[2];
    const double ux=b[0]-a[0],uy=b[1]-a[1],vx=c[0]-a[0],vy=c[1]-a[1];
    const double area=ux*vy-vx*uy;
    if(std::abs(area)<=1e-16) return false;
    const double rx=px-a[0],ry=py-a[1],u=(rx*vy-ry*vx)/area,v=(ux*ry-uy*rx)/area;
    if(std::min({u,v,1-u-v}) < -1.e-10) return false;
    z=a[2]+u*(b[2]-a[2])+v*(c[2]-a[2]);
    return true;
}
}

// Outputs per ray: status, crossed cells, interface slot, deep slot, first invalid cell.
// Coordinates record the outer, interface and deep heights and minimum posed/reference J.
// Ray/triangle coordinate in metres along the supplied unit director.
static bool director_height(const double* x,const std::array<int,3>& f,const double* o,
                            const double* d,double& hit) {
    const double *a=x+3*f[0],*b=x+3*f[1],*c=x+3*f[2];
    const std::array<double,3> ab={b[0]-a[0],b[1]-a[1],b[2]-a[2]};
    const std::array<double,3> ac={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
    const std::array<double,3> u={o[0]-a[0],o[1]-a[1],o[2]-a[2]};
    const std::array<double,3> dir={d[0],d[1],d[2]};
    auto cross=[](auto a,auto b){return std::array<double,3>{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};};
    auto dot=[](auto a,auto b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
    auto p=cross(dir,ac);double det=dot(ab,p);
    if(std::abs(det)<1.e-18)return false;
    auto q=cross(u,ab);double v=dot(u,p)/det,w=dot(dir,q)/det;
    if(v < -1.e-10 || w < -1.e-10 || v+w>1.+1.e-10)return false;
    hit=o[2]+dot(ac,q)/det;return std::isfinite(hit);
}
static int trace_columns_impl(const double* source,const double* posed,int nodes,
    const int* cells,const int* material,int count,const int* layers,const int* outer,int faces,
    const int* first,const double* xy,const double* z0,int rays,int* info,double* heights,double* material_length,int* interval,const double* direction=nullptr,double* incidence=nullptr,const Key* interfaces=nullptr,int interface_count=0) {
    if(nodes<=0 || nodes>=(1<<20) || count<=0 || count>100000000 || rays<=0 || faces<=0) return 1;
    std::vector<Face> sorted;sorted.reserve(4ULL*count);
    std::vector<int> neighbor(4ULL*count,-1);
    std::vector<double> jacobian(count);
    for(int t=0;t<count;++t) {
        const int* c=cells+4*t;
        for(int j=0;j<4;++j) if(c[j]<0||c[j]>=nodes) return 2;
        if(material[t]!=15 && material[t]!=16) return 3;
        const double ref=det(source,c),live=det(posed,c);
        if(!(ref>0) || !std::isfinite(live)) return 4;
        jacobian[t]=live/ref;
        for(int j=0;j<4;++j) {auto f=face(c,j);sorted.push_back({key(f[0],f[1],f[2]),4*t+j});}
    }
    std::sort(sorted.begin(),sorted.end(),[](Face a,Face b){return a.key<b.key;});
    for(size_t i=0;i<sorted.size();) {
        size_t end=i+1;while(end<sorted.size()&&sorted[end].key==sorted[i].key) ++end;
        if(end==i+2) {auto a=sorted[i].slot,b=sorted[i+1].slot;neighbor[a]=b;neighbor[b]=a;}
        if(end>i+2) for(size_t j=i;j<end;++j) neighbor[sorted[j].slot]=-2;
        i=end;
    }
    std::vector<int> start(faces,-1);
    for(int i=0;i<faces;++i) {
        const int* f=outer+3*i;
        for(int j=0;j<3;++j) if(f[j]<0||f[j]>=nodes || layers[f[j]]!=1) return 5;
        const auto k=key(f[0],f[1],f[2]);
        auto it=std::lower_bound(sorted.begin(),sorted.end(),k,[](Face a,Key b){return a.key<b;});
        if(it!=sorted.end()&&it->key==k&&neighbor[it->slot]==-1) start[i]=it->slot;
    }
    for(int i=0;i<rays;++i) {
        int* result=info+5*i; double* h=heights+4*i;
        std::fill_n(result,5,-1);std::fill_n(h,4,std::numeric_limits<double>::quiet_NaN());
        if(first[i]<0){result[0]=1;continue;}
        if(first[i]>=faces)return 6;
        int entered=start[first[i]];
        if(entered<0||material[entered/4]!=15){result[0]=2;continue;}
        if(incidence)incidence[i]=1.;
        const double origin[3]={xy[2*i],xy[2*i+1],z0[i]};
        double z=z0[i]; double lengths[2]={0.,0.}; int transitions=0;
        if(interval) { std::fill_n(interval+3*i,3,-1); std::fill_n(material_length+2*i,2,0.); }
        h[0]=z;h[3]=std::numeric_limits<double>::infinity();
        for(int it=1;;++it){
            const int t=entered/4;result[1]=it;result[4]=t;h[3]=std::min(h[3],jacobian[t]);
            if(!(jacobian[t]>0)){result[0]=3;break;}
            const int* c=cells+4*t; int exit=-1; double next=std::numeric_limits<double>::infinity();
            bool edge=false;
            for(int j=0;j<4;++j) {
                if(j==entered%4)continue;
                double hit;
                bool intersects=direction?director_height(posed,face(c,j),origin,direction+3*i,hit):height(posed,face(c,j),origin[0],origin[1],hit);
                if(!intersects || hit<=z+1.e-11)continue;
                if(std::abs(hit-next)<=1e-11)edge=true;
                else if(hit<next){next=hit;exit=4*t+j;edge=false;}
            }
            if(exit<0||edge){result[0]=4;break;}
            lengths[material[t]-15]+=next-z;
            const auto f=face(c,exit%4);
            auto isLayer=[&](int n){return layers[f[0]]==n&&layers[f[1]]==n&&layers[f[2]]==n;};
            const int other=neighbor[exit];
            if(interval) {
                interval[3*i]=transitions;
                interval[3*i+1]=isLayer(1)?1:isLayer(2)?2:isLayer(3)?3:0;
                interval[3*i+2]=material[t];
                std::copy_n(lengths,2,material_length+2*i);
            }
            if(other<0){
                if(other==-1 && isLayer(3) && material[t]==16 && std::isfinite(h[1])){
                    result[0]=0;result[3]=exit;result[4]=-1;h[2]=next;
                }else result[0]=5;
                break;
            }
            if(material[t]!=material[other/4]){
                if(!isLayer(2) || (!interval && (material[t]!=15 || std::isfinite(h[1])))){result[0]=6;break;}
                if(interfaces && !std::binary_search(interfaces,interfaces+interface_count,key(f[0],f[1],f[2]))){result[0]=6;break;}
                if(incidence){
                    const double *a=posed+3*f[0],*b=posed+3*f[1],*c=posed+3*f[2];
                    double u[3],v[3];for(int k=0;k<3;k++){u[k]=b[k]-a[k];v[k]=c[k]-a[k];}
                    const double nx=u[1]*v[2]-u[2]*v[1],ny=u[2]*v[0]-u[0]*v[2],nz=u[0]*v[1]-u[1]*v[0];
                    const double norm=std::sqrt(nx*nx+ny*ny+nz*nz);
                    if(!(norm>0.)){result[0]=4;break;}
                    incidence[i]=std::min(incidence[i],std::abs(nz)/norm);
                }
                ++transitions;
                if(!std::isfinite(h[1])){h[1]=next;result[2]=exit;}
            }
            if(it>=count){result[0]=7;break;}
            z=next;entered=other;
        }
    }
    return 0;
}

extern "C" int trace_registered_columns(const double* source,const double* posed,int nodes,
    const int* cells,const int* material,int count,const int* layers,const int* outer,int faces,
    const int* first,const double* xy,const double* z0,int rays,int* info,double* heights) {
    return trace_columns_impl(source,posed,nodes,cells,material,count,layers,outer,faces,first,xy,z0,rays,info,heights,nullptr,nullptr);
}
extern "C" int trace_material_intervals(const double* source,const double* posed,int nodes,
    const int* cells,const int* material,int count,const int* layers,const int* outer,int faces,
    const int* first,const double* xy,const double* z0,int rays,int* info,double* heights,double* lengths,int* intervals) {
    if(!lengths || !intervals)return 7;
    return trace_columns_impl(source,posed,nodes,cells,material,count,layers,outer,faces,first,xy,z0,rays,info,heights,lengths,intervals);
}

extern "C" int trace_material_directors(const double* source,const double* posed,int nodes,
    const int* cells,const int* material,int count,const int* layers,const int* outer,int faces,
    const int* first,const double* xy,const double* z0,int rays,int* info,double* heights,double* lengths,int* intervals,const double* directions) {
    if(!lengths || !intervals || !directions)return 7;
    for(int i=0;i<rays;i++){
        auto d=directions+3*i;
        if(!std::isfinite(d[0]+d[1]+d[2]) || std::abs(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]-1.)>1.e-7)return 8;
    }
    return trace_columns_impl(source,posed,nodes,cells,material,count,layers,outer,faces,first,xy,z0,rays,info,heights,lengths,intervals,directions);
}

extern "C" int trace_qualified_intervals(const double* source,const double* posed,int nodes,
    const int* cells,const int* material,int count,const int* layers,const int* outer,int faces,
    const int* first,const double* xy,const double* z0,int rays,int* info,double* heights,double* lengths,int* intervals,
    double* incidence,const Key* interfaces,int interface_count) {
    if(!lengths || !intervals || !incidence || !interfaces || interface_count<=0)return 7;
    return trace_columns_impl(source,posed,nodes,cells,material,count,layers,outer,faces,first,xy,z0,rays,info,heights,lengths,intervals,nullptr,incidence,interfaces,interface_count);
}
