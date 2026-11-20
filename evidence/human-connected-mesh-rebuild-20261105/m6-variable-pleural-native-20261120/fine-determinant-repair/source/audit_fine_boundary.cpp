#include "NumiHumanCoarseTissuePreparation.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>

namespace tissue = metalrobo::human::coarse_tissue;
using namespace tissue::registered_surface_intersection_detail;
using tissue::Face;
using tissue::V3;

struct Box {
    V3 low{INFINITY,INFINITY,INFINITY};
    V3 high{-INFINITY,-INFINITY,-INFINITY};
    void add(const V3& p) {
        for (unsigned i=0;i<3;++i) {
            low[i]=std::min(low[i],p[i]);
            high[i]=std::max(high[i],p[i]);
        }
    }
    bool overlaps(const Box& b) const {
        for (unsigned i=0;i<3;++i)
            if (high[i]<b.low[i] || b.high[i]<low[i]) return false;
        return true;
    }
};

struct Leaf {
    Triangle points;
    Face ids;
    unsigned role;
    bool moved;
    Box box;
};
struct TreeNode {
    Box box;
    unsigned first,last,left=0,right=0;
};
class Audit {
    std::vector<Leaf> faces;
    std::vector<unsigned> ids;
    std::vector<TreeNode> tree;
    std::array<std::array<std::uint64_t,4>,4> hits{};
    unsigned build(unsigned first,unsigned last) {
        TreeNode n;
        n.first=first;
        n.last=last;
        for (unsigned k=first;k<last;++k) {
            n.box.add(faces[ids[k]].box.low);
            n.box.add(faces[ids[k]].box.high);
        }
        const unsigned index=unsigned(tree.size());
        tree.push_back(n);
        if (last-first<=8) return index;
        unsigned axis=0;
        for (unsigned i=1;i<3;++i)
            if (n.box.high[i]-n.box.low[i]>n.box.high[axis]-n.box.low[axis]) axis=i;
        const unsigned mid=(first+last)/2;
        std::nth_element(ids.begin()+first,ids.begin()+mid,ids.begin()+last,[&](unsigned a,unsigned b) {
            const auto& x=faces[a].box;
            const auto& y=faces[b].box;
            return x.low[axis]+x.high[axis]<y.low[axis]+y.high[axis];
        });
        const unsigned l=build(first,mid);
        const unsigned r=build(mid,last);
        tree[index].left=l;
        tree[index].right=r;
        return index;
    }
public:
    explicit Audit(std::vector<Leaf> f) : faces(std::move(f)), ids(faces.size()) {
        std::iota(ids.begin(),ids.end(),0);
        tree.reserve(ids.size()/2);
        build(0,unsigned(ids.size()));
    }
    void compare(unsigned first,unsigned second,bool changesOnly) {
        const auto& a=tree[first];
        const auto& b=tree[second];
        if (!a.box.overlaps(b.box)) return;
        if (a.left && first==second) {
            compare(a.left,a.left,changesOnly);
            compare(a.left,a.right,changesOnly);
            compare(a.right,a.right,changesOnly);
            return;
        }
        if (!a.left && !b.left) {
            for (unsigned i=a.first;i<a.last;++i) {
                for (unsigned j=first==second ? i+1 : b.first;j<b.last;++j) {
                    const auto& x=faces[ids[i]];
                    const auto& y=faces[ids[j]];
                    if (changesOnly && !x.moved && !y.moved) continue;
                    if (!x.box.overlaps(y.box) ||
                        !intersectsBeyondSharedFeatures(x.points,x.ids,y.points,y.ids)) continue;
                    auto& count=hits[std::min(x.role,y.role)][std::max(x.role,y.role)];
                    ++count;
                    if (count<=24) {
                        std::cout<<"witness "<<x.role<<" "<<y.role<<" "<<ids[i]<<" "<<ids[j];
                        for (auto index:x.ids) std::cout<<" "<<index;
                        for (auto index:y.ids) std::cout<<" "<<index;
                        std::cout<<"\n";
                    }
                }
            }
            return;
        }
        if (a.left && (!b.left || a.last-a.first>=b.last-b.first)) {
            compare(a.left,second,changesOnly);
            compare(a.right,second,changesOnly);
        } else {
            compare(first,b.left,changesOnly);
            compare(first,b.right,changesOnly);
        }
    }
    void report() const {
        for (unsigned i=0;i<4;++i)
            for (unsigned j=i;j<4;++j)
                std::cout<<"pair "<<i<<" "<<j<<" "<<hits[i][j]<<"\n";
    }
};
int main(int argc,char** argv) {
    if (argc!=2) return 2;
    std::ifstream in(argv[1],std::ios::binary);
    const auto read=[&]<class T>(T& v) -> bool {
        return bool(in.read(reinterpret_cast<char*>(&v),sizeof(v)));
    };
    std::uint32_t np,nf;
    if (!read(np)||!read(nf)||np!=311020||nf>2'000'000) return 3;
    std::vector<std::array<float,3>> old(np),now(np);
    for (auto& p:old) if (!read(p)) return 4;
    for (auto& p:now) if (!read(p)) return 4;
    std::vector<std::array<std::uint32_t,4>> ids(nf);
    for (auto& face:ids)
        if (!read(face) || face[3]>=4 || std::max({face[0],face[1],face[2]})>=np) return 5;
    for (unsigned stage=0;stage<2;++stage) {
        const auto& points=stage ? now : old;
        std::vector<Leaf> bank;
        bank.reserve(nf);
        for (const auto& f:ids) {
            Leaf leaf{};
            leaf.role=f[3];
            for (unsigned k=0;k<3;++k) {
                const auto node=f[k];
                leaf.ids[k]=node;
                leaf.points[k]={points[node][0],points[node][1],points[node][2]};
                leaf.box.add(leaf.points[k]);
                leaf.moved|=(old[node]!=now[node]);
            }
            bank.push_back(leaf);
        }
        Audit audit(std::move(bank));
        std::cout<<"stage "<<stage<<"\n";
        audit.compare(0,0,false);
        audit.report();
    }
    return 0;
}
