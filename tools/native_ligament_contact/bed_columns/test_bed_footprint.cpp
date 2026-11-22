#include "NumiHumanBedFootprint.hpp"
#include <numeric>
#include <iostream>
using namespace numiHumanBedFootprint;
static void close(double a,double b) { if (std::abs(a-b)>2e-12) throw std::runtime_error("wrong projected material area"); }
int main() {
    std::vector<Point> p={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}};
    const std::vector<unsigned> f={0,1,2,5,4,3};
    std::vector<double> limit(6,2);
    auto check=[&](const std::vector<double>& w){close(std::accumulate(w.begin(),w.end(),0.),.5);};
    auto a=areas(p,f,limit); check(a);
    for (unsigned i=0;i<3;++i) {close(a[i],1./6);close(a[i+3],0);}
    // A crossing sheet z=x-1/2 lies below the flat sheet for x<1/2.
    for (unsigned i=3;i<6;++i) p[i].z=p[i].x-.5;
    a=areas(p,f,limit); check(a); close(a[0]+a[1]+a[2],.125);
    close(a[1],1./12); close(a[0],1./48);
    auto other=areas(p,{3,4,5,2,1,0},limit);
    for (unsigned i=0;i<6;++i) close(a[i],other[i]);
    for(unsigned i=3;i<6;++i)p[i].z=0.;
    a=areas(p,f,limit);check(a);close(a[3]+a[4]+a[5],0);
    // A height cutoff slices the triangle interior and integrates barycentric weights.
    p[1].z=1.;a=areas(p,{0,1,2},std::vector<double>(6,.5));
    close(a[0]+a[1]+a[2],.375);close(a[1],1./12);
    // A partly overlapping sheet, shared-edge surface and a vertical face.
    p={{0,0,0},{1,0,0},{0,1,0},{1,1,0},{.5,0,-1},{1.5,0,-1},{.5,1,-1},{0,0,1}};
    a=areas(p,{0,1,2,1,3,2,4,5,6,0,7,2},std::vector<double>(p.size(),2));
    close(std::accumulate(a.begin(),a.end(),0.),1.125);
    close(a[4]+a[5]+a[6],.5);close(a[7],0.);
    // Two intersecting lower projected strips cover the unit square once.
    // Adding a deeper copy of either strip must replace only its own share.
    std::vector<Point> rectangles;
    std::vector<unsigned> triangles;
    auto rectangle=[&](double x0, double y0, double x1, double y1, double z) {
        unsigned k=static_cast<unsigned>(rectangles.size());
        rectangles.insert(rectangles.end(), {{x0,y0,z},{x1,y0,z},{x1,y1,z},{x0,y1,z}});
        triangles.insert(triangles.end(), {k,k+1,k+2,k,k+2,k+3});
    };
    rectangle(0,0,1,1,2);
    rectangle(.25,0,.75,1,1);
    rectangle(0,.25,1,.75,0);
    rectangle(.25,0,.75,1,-1);
    const auto coverage=areas(rectangles,triangles,std::vector<double>(rectangles.size(),2));
    for (unsigned i=0; i<4; ++i) {
        const double expected[]={.25,0,.25,.5};
        close(std::accumulate(coverage.begin()+4*i, coverage.begin()+4*(i+1),0.),expected[i]);
    }
    std::cout<<"PASS projected-area, depth crossing, coplanar ties, orientation, height cut, multiple regions and vertical faces\n";
}
