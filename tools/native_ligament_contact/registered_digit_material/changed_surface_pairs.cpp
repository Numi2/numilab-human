// Enumerate exact triangle intersections involving an edited material face.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
using U = std::uint32_t;
using V = std::array<double, 3>;
using F = std::array<U, 3>;
using Pair = std::array<U, 2>;
extern "C" void batch(const double*, const unsigned*, const unsigned*, unsigned, unsigned char*);
using Box = std::array<V, 2>;
Box empty() {
    const auto inf = std::numeric_limits<double>::infinity();
    return {{ {inf,inf,inf}, {-inf,-inf,-inf} }};
}
void add(Box& a, const Box& b) {
    for (int k=0;k<3;k++) {
        a[0][k]=std::min(a[0][k],b[0][k]);
        a[1][k]=std::max(a[1][k],b[1][k]);
    }
}
bool overlaps(const Box& a, const Box& b) {
    for(int k=0;k<3;k++) if(a[1][k]<b[0][k] || b[1][k]<a[0][k]) return false;
    return true;
}
template<class T> std::vector<T> read(std::istream& in, size_t count) {
    std::vector<T> data(count);
    if (!in.read(reinterpret_cast<char*>(data.data()), count * sizeof(T))) throw std::runtime_error("Truncated surface input");
    return data;
}
int main(int argc,char** argv) {
    if(argc!=3) return 2;
    std::ifstream in(argv[1],std::ios::binary);
    std::ofstream out(argv[2],std::ios::binary);
    const auto sizes=read<U>(in,2);
    const auto x=read<V>(in,sizes[0]); const auto f=read<F>(in,sizes[1]);
    const auto edit=read<std::uint8_t>(in,f.size());
    if(!out || f.empty() || in.peek()!=std::char_traits<char>::eof()) return 3;
    std::vector<Box> bounds(f.size(),empty());
    for(size_t i=0;i<f.size();i++) for(U j:f[i]) { const auto& p=x.at(j); add(bounds[i],{p,p}); }
    std::vector<U> idx(f.size()); std::iota(idx.begin(),idx.end(),0);
    struct Node {Box bounds; U begin,end,left,right;};
    std::vector<Node> tree;
    std::function<U(U,U)> build=[&](U a,U b) {
        Box box=empty(); for(U k=a;k<b;k++) add(box,bounds[idx[k]]);
        const U n=tree.size(); tree.push_back({box,a,b,0,0});
        if (b-a>8) {
            int axis=0;
            for(int d=1;d<3;d++) if(box[1][d]-box[0][d]>box[1][axis]-box[0][axis]) axis=d;
            const U m=a+(b-a)/2;
            std::nth_element(idx.begin()+a,idx.begin()+m,idx.begin()+b,[&](U i,U j){
                return bounds[i][0][axis]+bounds[i][1][axis] < bounds[j][0][axis]+bounds[j][1][axis];
            });
            const U left=build(a,m),right=build(m,b);
            tree[n].left=left; tree[n].right=right;
        }
        return n;
    };
    build(0,f.size());
    std::vector<Pair> pairs;
    auto flush=[&]{
        if(pairs.empty()) return;
        std::vector<unsigned char> hit(pairs.size());
        batch(x.front().data(),f.front().data(),pairs.front().data(),pairs.size(),hit.data());
        for(size_t k=0;k<hit.size();k++) if(hit[k]) out.write(reinterpret_cast<const char*>(&pairs[k]),sizeof(Pair));
        pairs.clear();
    };
    for(U i=0;i<f.size();i++) if(edit[i]) {
        std::vector<U> stack{0};
        while(!stack.empty()) {
            const Node& n=tree[stack.back()]; stack.pop_back();
            if(!overlaps(n.bounds,bounds[i])) continue;
            if(n.left) {stack.push_back(n.left);stack.push_back(n.right);continue;}
            for(U k=n.begin;k<n.end;k++) {
                U j=idx[k]; if(j==i || (edit[j] && j<i) || !overlaps(bounds[i],bounds[j])) continue;
                pairs.push_back({std::min(i,j),std::max(i,j)});
                if(pairs.size()==32768) flush();
            }
        }
    }
    flush();
    return out ? 0 : 4;
}
