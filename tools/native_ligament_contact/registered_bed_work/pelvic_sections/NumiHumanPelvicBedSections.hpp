#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// These are the bilateral posterior pelvic footprint seeds in the supported
// reference NHCNT. The skeleton owner of a seed and its skin donor weights
// are separate properties. The GPU continues to use the latter for work.
namespace numiPelvicBed {
struct Seed { unsigned parent, geometry, body; };
constexpr std::array<Seed,2> seeds={{{6,33529,1},{7,20347,128}}};
inline bool enabled() {
    const auto* flag=std::getenv("NUMI_HUMAN_PELVIC_BED_SECTIONS");
    if (!flag || !std::strcmp(flag,"0")) return false;
    if (!std::strcmp(flag,"1")) return true;
    throw std::invalid_argument("Invalid pelvic bed section option");
}

template<class Value>
double weightedCut(std::vector<unsigned> loaded,const std::vector<double>& areas,Value value) {
    if (loaded.size()<2) throw std::domain_error("Insufficient posterior bed quadrature");
    std::sort(loaded.begin(),loaded.end(),[&](unsigned a,unsigned b) {
        return value(a)==value(b) ? a<b : value(a)<value(b);
    });
    double total=0;
    for(auto v:loaded) total+=areas[v];
    double cumulative=0,score=INFINITY,cut=0;
    for(size_t j=0;j+1<loaded.size();++j) {
        cumulative+=areas[loaded[j]];
        const double a=value(loaded[j]), b=value(loaded[j+1]);
        if(b>a && std::abs(cumulative-total/2)<score) {
            cut=(a+b)*.5;
            score=std::abs(cumulative-total/2);
        }
    }
    if(!std::isfinite(score)) throw std::domain_error("Degenerate posterior bed quadrature");
    return cut;
}

template<class Points>
void partition(const Points& points,const std::vector<double>& areas,
               const std::vector<unsigned>& sourceSeeds,std::vector<unsigned>& owner,
               std::vector<unsigned>& counts) {
    if(points.size()!=owner.size() || areas.size()!=owner.size() || counts.size()!=44)
        throw std::invalid_argument("Incorrect supported tissue domain");
    for(unsigned side=0;side<seeds.size();++side) {
        const auto parent=seeds[side].parent;
        std::vector<unsigned> vertices,loaded;
        for(unsigned i=0;i<owner.size();++i) if(owner[i]==parent) {
            vertices.push_back(i);
            if(areas[i]>0) loaded.push_back(i);
        }
        // First separate eminences which reach the plane early from material
        // higher on its sides. Split each stratum along its principal bed axis.
        const auto high=[&](unsigned i){return double(points[i].z);};
        const double zcut=weightedCut(loaded,areas,high);
        std::array<std::array<double,3>,2> lines{};
        for(unsigned layer=0;layer<2;++layer) {
            std::vector<unsigned> group;
            double sum=0,mx=0,my=0;
            for(auto i:loaded) if((high(i)>zcut)==bool(layer)) {
                group.push_back(i);sum+=areas[i];mx+=areas[i]*points[i].x;my+=areas[i]*points[i].y;
            }
            if(!(sum>0))throw std::domain_error("Empty pelvic support stratum");
            mx/=sum;my/=sum;double xx=0,xy=0,yy=0;
            for(auto i:group) {
                const double dx=points[i].x-mx,dy=points[i].y-my;
                xx+=areas[i]*dx*dx;xy+=areas[i]*dx*dy;yy+=areas[i]*dy*dy;
            }
            const double angle=.5*std::atan2(2*xy,xx-yy),nx=std::cos(angle),ny=std::sin(angle);
            const auto along=[&](unsigned i){return nx*points[i].x+ny*points[i].y;};
            lines[layer]={nx,ny,weightedCut(group,areas,along)};
        }
        const auto bin=[&](unsigned i) {
            const unsigned layer=high(i)>zcut;
            const auto l=lines[layer];
            return 2*layer+unsigned(l[0]*points[i].x+l[1]*points[i].y>l[2]);
        };
        const unsigned seed=sourceSeeds.at(parent);
        if(owner.at(seed)!=parent) throw std::domain_error("Pelvic source seed moved outside its region");
        std::array<unsigned,4> rows;
        unsigned next=38+3*side;
        for(unsigned k=0;k<4;++k) rows[k]=(k==bin(seed))?parent:next++;
        counts[parent]=0;
        for(auto i:vertices) {owner[i]=rows[bin(i)];++counts[owner[i]];}
    }
}
}
