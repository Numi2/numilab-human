#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// Project the resting skin onto the physical bed. At a given bed coordinate
// only the lowest triangle owns the projected area. Integrate its linear
// vertex weights, so adjacent anatomical sections share the area exactly.
namespace numiHumanBedFootprint {
struct Point { double x, y, z; };
struct UV { double u, v; };
using Polygon = std::vector<UV>;
using Line = std::array<double, 3>; // values at the three subject corners

inline double value(const Line& h, UV p) {
    return h[0] + (h[1] - h[0])*p.u + (h[2] - h[0])*p.v;
}

inline Polygon clip(const Polygon& poly, const Line& h, bool positive) {
    Polygon result;
    if (poly.empty()) return result;
    UV a = poly.back();
    double fa = value(h, a);
    for (const auto b : poly) {
        const double fb = value(h, b);
        if ((fa > 0 && fb < 0) || (fa < 0 && fb > 0)) {
            const double t = fa/(fa-fb);
            result.push_back({a.u+t*(b.u-a.u), a.v+t*(b.v-a.v)});
        }
        if (positive ? fb >= 0 : fb <= 0) result.push_back(b);
        a=b; fa=fb;
    }
    return result;
}

inline std::array<double,3> weights(const Polygon& poly) {
    std::array<double,3> w{};
    for (std::size_t i=2; i<poly.size(); ++i) {
        const auto a=poly[0], b=poly[i-1], c=poly[i];
        const double twice=(b.u-a.u)*(c.v-a.v)-(b.v-a.v)*(c.u-a.u);
        if (twice<=0) continue;
        const double u=a.u+b.u+c.u, v=a.v+b.v+c.v;
        w[0]+=twice*(3-u-v)/6.; w[1]+=twice*u/6.; w[2]+=twice*v/6.;
    }
    return w;
}

struct Face {
    std::array<unsigned,3> vertex;
    std::array<Point,3> p;
    double det, xmin, xmax, ymin, ymax, zmin, zmax;
};

inline double orient(Point a, Point b, Point c) {
    return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}

// The four positive half-spaces form the part of A hidden by B.
// Depth uses a determinant instead of division by B's projected area.
inline bool occluder(const Face& a, const Face& b, bool bWinsTie, std::array<Line,4>& h) {
    if (a.zmax < b.zmin || a.xmax<=b.xmin || b.xmax<=a.xmin ||
        a.ymax<=b.ymin || b.ymax<=a.ymin) return false;
    const double sign=b.det>0 ? 1. : -1.;
    for (unsigned edge=0; edge<3; ++edge) {
        for (unsigned k=0; k<3; ++k)
            h[edge][k]=sign*orient(b.p[edge], b.p[(edge+1)%3], a.p[k]);
        if (*std::max_element(h[edge].begin(), h[edge].end())<=0) return false;
    }
    const auto p=b.p[0], q=b.p[1], r=b.p[2];
    const double nx=(q.y-p.y)*(r.z-p.z)-(q.z-p.z)*(r.y-p.y);
    const double ny=(q.z-p.z)*(r.x-p.x)-(q.x-p.x)*(r.z-p.z);
    for (unsigned k=0; k<3; ++k) {
        const auto s=a.p[k];
        const double x=nx*(s.x-p.x), y=ny*(s.y-p.y), z=b.det*(s.z-p.z);
        h[3][k]=sign*(x+y+z);
        if (std::abs(h[3][k]) <= 16*std::numeric_limits<double>::epsilon()*
                                     (std::abs(x)+std::abs(y)+std::abs(z))) h[3][k]=0.;
    }
    if (*std::max_element(h[3].begin(),h[3].end())<=0) {
        if (!bWinsTie || *std::min_element(h[3].begin(),h[3].end())<0) return false;
        h[3].fill(1.);
    }
    return true;
}

inline void subtract(std::vector<Polygon>& fragments, const std::array<Line,4>& h) {
    std::vector<Polygon> next;
    for (auto& fragment : fragments) {
        Polygon common=fragment;
        for (const auto& line : h) common=clip(common,line,true);
        auto w=weights(common);
        if (w[0]+w[1]+w[2]<=0.) { next.push_back(std::move(fragment)); continue; }
        for (const auto& line : h) {
            Polygon outside=clip(fragment,line,false);
            if (outside.size()>=3) next.push_back(std::move(outside));
            fragment=clip(fragment,line,true);
        }
    }
    fragments=std::move(next);
}

inline std::vector<double> areas(const std::vector<Point>& points, const std::vector<unsigned>& indices,
                                 const std::vector<double>& heightLimit) {
    if (indices.empty() || indices.size()%3 || points.size()!=heightLimit.size())
        throw std::runtime_error("invalid bed projection domain");
    for (std::size_t i=0;i<points.size();++i) {
        const auto p=points[i];
        if (!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||!std::isfinite(heightLimit[i]))
            throw std::runtime_error("nonfinite skin or support section");
    }
    std::vector<Face> faces;
    for (std::size_t i=0;i<indices.size();i+=3) {
        Face f{}; for (unsigned k=0;k<3;++k) {f.vertex[k]=indices[i+k]; f.p[k]=points.at(f.vertex[k]);}
        const auto a=f.p[0], b=f.p[1], c=f.p[2]; f.det=orient(a,b,c);
        if (f.det==0.) continue;
        f.xmin=std::min({a.x,b.x,c.x}); f.xmax=std::max({a.x,b.x,c.x});
        f.ymin=std::min({a.y,b.y,c.y}); f.ymax=std::max({a.y,b.y,c.y});
        f.zmin=std::min({a.z,b.z,c.z}); f.zmax=std::max({a.z,b.z,c.z});
        faces.push_back(f);
    }
    if (faces.empty()) throw std::runtime_error("skin has no bed projection");
    double xmin=faces[0].xmin,xmax=faces[0].xmax,ymin=faces[0].ymin,ymax=faces[0].ymax;
    for (const auto& f : faces) {
        xmin=std::min(xmin,f.xmin); xmax=std::max(xmax,f.xmax);
        ymin=std::min(ymin,f.ymin); ymax=std::max(ymax,f.ymax);
    }
    const double width=std::sqrt((xmax-xmin)*(ymax-ymin)/faces.size());
    const int nx=std::clamp(int(std::ceil((xmax-xmin)/width)),1,1024);
    const int ny=std::clamp(int(std::ceil((ymax-ymin)/width)),1,1024);
    auto bins=[&](const Face& f) {
        auto ix=[&](double x){return std::clamp(int((x-xmin)/(xmax-xmin)*nx),0,nx-1);};
        auto iy=[&](double y){return std::clamp(int((y-ymin)/(ymax-ymin)*ny),0,ny-1);};
        return std::array<int,4>{ix(f.xmin),ix(f.xmax),iy(f.ymin),iy(f.ymax)};
    };
    std::vector<std::vector<unsigned>> grid(nx*ny);
    for (unsigned i=0;i<faces.size();++i) {
        const auto box=bins(faces[i]);
        for (int y=box[2];y<=box[3];++y) for(int x=box[0];x<=box[1];++x) grid[y*nx+x].push_back(i);
    }
    std::vector<unsigned> seen(faces.size(),std::numeric_limits<unsigned>::max());
    std::vector<double> result(points.size());
    for (unsigned i=0;i<faces.size();++i) {
        const auto& a=faces[i];
        if (a.zmin>=std::max({heightLimit[a.vertex[0]],heightLimit[a.vertex[1]],heightLimit[a.vertex[2]]})
            && a.zmax!=a.zmin) continue;
        std::vector<Polygon> visible={{{0,0},{1,0},{0,1}}};
        const auto box=bins(a);
        for(int y=box[2];y<=box[3]&&!visible.empty();++y) for(int x=box[0];x<=box[1]&&!visible.empty();++x) {
            for(auto j:grid[y*nx+x]) {
                if(j==i || seen[j]==i) continue;
                seen[j]=i;
                std::array<Line,4> h;
                if(occluder(a,faces[j],j<i,h)) subtract(visible,h);
                if(visible.empty()) break;
            }
        }
        for (const auto& part:visible) for(unsigned k=0;k<3;++k) {
            const double limit=heightLimit[a.vertex[k]];
            const Line band={limit-a.p[0].z,limit-a.p[1].z,limit-a.p[2].z};
            result[a.vertex[k]]+=std::abs(a.det)*weights(clip(part,band,true))[k];
        }
    }
    return result;
}
} // namespace numiHumanBedFootprint
