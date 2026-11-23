// Lowest registered boundary at each fixed material quadrature point.
// The face and barycentric coordinates identify the contributing source nodes.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" int bed_boundary_columns(const double* xyz, int nodes,
    const int32_t* tri, int faces, const double* xy, int rays,
    int32_t* owners, double* z, double* bary) {
    if (!xyz || !tri || !xy || !owners || !z || !bary || nodes<=0 || faces<=0 || rays<=0) return 1;
    constexpr double width=0.01;
    std::array<double,2> lower{xy[0],xy[1]}, upper=lower;
    for(int i=0;i<rays;++i) for(int a=0;a<2;++a) {
        const double x=xy[2*i+a];
        if(!std::isfinite(x)) return 2;
        lower[a]=std::min(lower[a],x);upper[a]=std::max(upper[a],x);
    }
    const int nx=int((upper[0]-lower[0])/width)+1, ny=int((upper[1]-lower[1])/width)+1;
    if(nx<=0 || ny<=0 || int64_t(nx)*ny>1000000) return 3;
    const auto bin=[&](double v,int a){return std::clamp(int(std::floor((v-lower[a])/width)),0,(a?ny:nx)-1);};
    std::vector<std::vector<int>> bins(size_t(nx)*ny);
    for(int i=0;i<rays;++i) {
        bins[bin(xy[2*i+1],1)*nx+bin(xy[2*i],0)].push_back(i);
        owners[i]=-1;z[i]=std::numeric_limits<double>::infinity();
        std::fill_n(bary+3*i,3,0.);
    }
    for(int f=0;f<faces;++f) {
        const int32_t* ids=tri+3*f;
        for(int d=0;d<3;++d) if(ids[d]<0 || ids[d]>=nodes) return 4;
        const double *a=xyz+3*ids[0],*b=xyz+3*ids[1],*c=xyz+3*ids[2];
        const double bx=b[0]-a[0],by=b[1]-a[1],cx=c[0]-a[0],cy=c[1]-a[1];
        const double det=bx*cy-by*cx;
        if(std::abs(det)<=1e-16) continue;
        const double xmin=std::min({a[0],b[0],c[0]}), xmax=std::max({a[0],b[0],c[0]});
        const double ymin=std::min({a[1],b[1],c[1]}), ymax=std::max({a[1],b[1],c[1]});
        if(xmax<lower[0] || xmin>upper[0] || ymax<lower[1] || ymin>upper[1]) continue;
        for(int iy=bin(ymin,1);iy<=bin(ymax,1);++iy) for(int ix=bin(xmin,0);ix<=bin(xmax,0);++ix) {
            for(int i:bins[iy*nx+ix]) {
                const double x=xy[2*i],y=xy[2*i+1];
                if(x<xmin || x>xmax || y<ymin || y>ymax) continue;
                const double u=((x-a[0])*cy-(y-a[1])*cx)/det;
                const double v=(bx*(y-a[1])-by*(x-a[0]))/det;
                if(u< -1e-9 || v< -1e-9 || u+v>1.+1e-9) continue;
                const double h=a[2]+u*(b[2]-a[2])+v*(c[2]-a[2]);
                if(h>=z[i]) continue;
                z[i]=h;owners[i]=f;bary[3*i]=1-u-v;bary[3*i+1]=u;bary[3*i+2]=v;
            }
        }
    }
    return 0;
}
