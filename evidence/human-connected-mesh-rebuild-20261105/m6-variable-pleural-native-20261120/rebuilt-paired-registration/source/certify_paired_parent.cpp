#include "NumiHumanCoarseTissuePreparation.hpp"
#include "NumiHumanElbowExplicitPairedTopology008.hpp"
#include <fstream>
#include <iostream>

namespace tissue = metalrobo::human::coarse_tissue;
using tissue::Face;
using tissue::V3;
using tissue::dot;
using tissue::sub;
using namespace tissue::registered_surface_intersection_detail;

struct Bounds {
    Triangle t;
    Face ids;
    V3 lower;
    V3 upper;
    Bounds(const std::vector<V3>& points, Face ids, unsigned offset) : ids(ids) {
        for (unsigned k=0; k!=3; ++k) {
            t[k]=points[ids[k]];
            this->ids[k]+=offset;
        }
        for (unsigned k=0; k!=3; ++k) {
            lower[k]=std::min({t[0][k],t[1][k],t[2][k]});
            upper[k]=std::max({t[0][k],t[1][k],t[2][k]});
        }
    }
    bool overlaps(const Bounds& other) const {
        for (unsigned k=0; k!=3; ++k)
            if (lower[k]>other.upper[k] || other.lower[k]>upper[k]) return false;
        return true;
    }
};

int main(int argc, char** argv) {
    if (argc!=2) return 2;
    std::ifstream stream(argv[1]);
    constexpr unsigned rows=2411, nf=4777;
    std::array<std::vector<V3>,3> layers;
    std::vector<V3> child, parent;
    std::vector<Face> faces(nf);
    for (unsigned i=0; i!=layers.size(); ++i) {
        layers[i].resize(rows);
        for (unsigned j=0; j!=rows; ++j) {
            auto& point=layers[i][j];
            if (!(stream>>point[0]>>point[1]>>point[2])) return 3;
            child.push_back(point);
            if (j<tissue::kElbowRefinementParentVertexCount) parent.push_back(point);
        }
    }
    for (auto& f: faces)
        if (!(stream>>f[0]>>f[1]>>f[2]) || std::any_of(f.begin(), f.end(),
                [](unsigned index) { return index>=rows; })) return 4;
    std::string failure;
    const bool ancestry=tissue::validateExplicitParentTetPartition(
        child, tissue::kElbowRefinementChildTetrahedra, tissue::kElbowRefinementChildMaterialCodes,
        tissue::kElbowRefinementChildParentTetrahedra, parent, tissue::kElbowRefinementParentTetrahedra,
        tissue::kElbowRefinementParentMaterialCodes, failure);
    std::cout<<"partition="<<(ancestry?"pass":"fail")<<" "<<failure<<'\n';
    bool pass=ancestry;
    std::array<std::vector<Bounds>,3> bounds;
    for (unsigned a=0; a!=3; ++a)
        for (auto f:faces) bounds[a].emplace_back(layers[a], f, a*rows);
    for (unsigned a=0; a!=3; ++a) {
        for (unsigned b=a; b!=3; ++b) {
            unsigned count=0;
            for (unsigned i=0; i!=nf; ++i) {
                const auto& x=bounds[a][i];
                for (unsigned j=(a==b?i+1:0); j!=nf; ++j) {
                    const auto& y=bounds[b][j];
                    if (!x.overlaps(y) || !intersectsBeyondSharedFeatures(x.t,x.ids,y.t,y.ids)) continue;
                    ++count;
                    if (count<=8) std::cout<<"crossing "<<a<<','<<b<<" faces="<<i<<','<<j<<'\n';
                }
            }
            pass&=(count==0);
            std::cout<<"layer_pair="<<a<<','<<b<<" crossings="<<count<<'\n';
        }
    }
    return pass?0:1;
}
