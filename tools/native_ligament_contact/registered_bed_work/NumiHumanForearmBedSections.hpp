#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

inline bool forearmBedSectionsRequested() {
    const char* v = std::getenv("NUMI_HUMAN_FOREARM_BED_SECTIONS");
    if (!v || !std::strcmp(v,"0")) return false;
    if (!std::strcmp(v,"1")) return true;
    throw std::invalid_argument("NUMI_HUMAN_FOREARM_BED_SECTIONS expects 0 or 1");
}

inline constexpr std::array<unsigned,2> forearmBedParents = {12,22};
inline constexpr std::array<unsigned,2> forearmBedSeedIds = {6890,47596};
inline constexpr std::array<unsigned,2> forearmBedBodies = {42,92};

// Partition an existing footprint along its longest in-plane principal axis.
// Every source column retains its area and material samples. The native
// contact solver receives four disjoint sets of material points per forearm.
template<class Points>
void partitionForearmBedColumns(const Points& points,const std::vector<double>& areas,
        const std::vector<unsigned>& seeds,std::vector<unsigned>& owner,std::vector<unsigned>& counts) {
    if(points.size()!=owner.size() || areas.size()!=owner.size() || counts.size()!=38u)
        throw std::invalid_argument("Invalid bilateral bed partition domain");
    for(unsigned side=0;side<forearmBedParents.size();++side) {
        const unsigned parent=forearmBedParents[side];
        std::vector<unsigned> vertices,loaded;
        double total=0,meanX=0,meanY=0;
        for(unsigned i=0;i<owner.size();++i) if(owner[i]==parent) {
            vertices.push_back(i);
            if(areas[i]>0) {
                loaded.push_back(i);
                total+=areas[i]; meanX+=areas[i]*points[i].x; meanY+=areas[i]*points[i].y;
            }
        }
        if(loaded.size()<4 || total<=0 || owner.at(seeds.at(parent))!=parent)
            throw std::domain_error("No quadrature support for parent seed");
        meanX/=total; meanY/=total;
        double xx=0,xy=0,yy=0;
        for(unsigned i:loaded) {
            double x=points[i].x-meanX,y=points[i].y-meanY;
            xx+=areas[i]*x*x; xy+=areas[i]*x*y; yy+=areas[i]*y*y;
        }
        const double ang=.5*std::atan2(2*xy,xx-yy);
        const double dx=std::cos(ang),dy=std::sin(ang);
        auto coordinate=[&](unsigned i){ return dx*(points[i].x-meanX)+dy*(points[i].y-meanY); };
        std::sort(loaded.begin(),loaded.end(),[&](unsigned a,unsigned b){
            const double x=coordinate(a),y=coordinate(b);return x!=y ? x<y : a<b;
        });
        std::array<double,3> cuts{};
        double area=0;unsigned cut=0;
        for(unsigned j=0;j+1<loaded.size();++j) {
            area+=areas[loaded[j]];
            if(cut<cuts.size() && area>=total*double(cut+1)*.25 &&
               coordinate(loaded[j+1])>coordinate(loaded[j])) {
                cuts[cut++]=.5*(coordinate(loaded[j])+coordinate(loaded[j+1]));
            }
        }
        if(cut!=cuts.size())throw std::domain_error("Insufficient distinct loaded columns");
        auto bin=[&](unsigned i){return unsigned(std::upper_bound(cuts.begin(),cuts.end(),coordinate(i))-cuts.begin());};
        std::array<unsigned,4> rows;unsigned next=32+3*side;
        const unsigned seedBin=bin(seeds[parent]);
        for(unsigned j=0;j<4;++j)rows[j]=j==seedBin ? parent:next++;
        counts[parent]=0;
        for(unsigned i:vertices){owner[i]=rows[bin(i)];++counts[owner[i]];}
    }
}
