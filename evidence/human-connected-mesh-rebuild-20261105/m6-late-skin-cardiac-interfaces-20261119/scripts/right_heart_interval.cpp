// Authoring test for the common-field cardiac deformation in native local metres.
// Bounds both Float32 fused and unfused evaluation of the seven basis terms.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using Vec = std::array<double, 3>;
using FVec = std::array<float, 3>;
using Face = std::array<uint32_t, 3>;
using Vertex = std::array<FVec, 8>;
using Coordinates = std::array<float, 7>;

static Vec sub(Vec a, const Vec& b) {
    for (int i = 0; i < 3; ++i) a[i] -= b[i];
    return a;
}
static Vec add(Vec a, const Vec& b) {
    for (int i = 0; i < 3; ++i) a[i] += b[i];
    return a;
}
static Vec scaled(Vec a, double factor) {
    for (auto& x : a) x *= factor;
    return a;
}
static double dot(const Vec& a, const Vec& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
static Vec cross(const Vec& a, const Vec& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
struct Interval {
    double lo, hi;
};
struct Range {
    Vec lo, hi;

    Vec center() const { return scaled(add(lo, hi), .5); }
    void include(const Range& rhs) {
        for (int i = 0; i < 3; ++i) {
            lo[i] = std::min(lo[i], rhs.lo[i]);
            hi[i] = std::max(hi[i], rhs.hi[i]);
        }
    }
    bool overlaps(const Range& rhs) const {
        for (int i = 0; i < 3; ++i)
            if (lo[i] > rhs.hi[i] || rhs.lo[i] > hi[i]) return false;
        return true;
    }
};
static double down(double x) {
    return std::nextafter(x, -std::numeric_limits<double>::infinity());
}
static double up(double x) {
    return std::nextafter(x, std::numeric_limits<double>::infinity());
}
static Interval add(Interval a, Interval b) {
    return {down(a.lo+b.lo), up(a.hi+b.hi)};
}
static Interval mul(Interval a, Interval b) {
    const std::array<double,4> terms{a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi};
    return {down(*std::min_element(terms.begin(),terms.end())),
            up(*std::max_element(terms.begin(),terms.end()))};
}
static Range difference(const Range& a, const Range& b) {
    Range out;
    for (int i = 0; i < 3; ++i) {
        out.lo[i] = down(a.lo[i] - b.hi[i]);
        out.hi[i] = up(a.hi[i] - b.lo[i]);
    }
    return out;
}
static Interval component(const Range& a, int i) {
    return {a.lo[i],a.hi[i]};
}
static Range cross(const Range& a, const Range& b) {
    Range out;
    for (int i = 0; i < 3; ++i) {
        const int j = (i+1)%3, k = (i+2)%3;
        const auto p = mul(component(a,j),component(b,k));
        const auto q = mul(component(a,k),component(b,j));
        out.lo[i] = down(p.lo-q.hi);
        out.hi[i] = up(p.hi-q.lo);
    }
    return out;
}
static Interval dot(const Range& a, const Range& b) {
    return add(add(mul(component(a,0),component(b,0)),
                   mul(component(a,1),component(b,1))),
               mul(component(a,2),component(b,2)));
}
static Interval dot(const Range& a, const Vec& b) {
    return dot(a, Range{b,b});
}
static Range pose(const Vertex& v, const Coordinates& lower, const Coordinates& upper,
                  Vec& rounding, bool fused) {
    Range out;
    for (int i = 0; i < 3; ++i) {
        float lo=v[0][i], hi=lo;
        for (int j = 0; j < 7; ++j) {
            const float a=v[j+1][i];
            const float x=a<0 ? upper[j]:lower[j], y=a<0 ? lower[j]:upper[j];
            const float termLo=a*x, termHi=a*y;
            // Monotonicity gives exact Float32 endpoint bounds for each multiply-add.
            auto halfUlp=[](float value) {
                const float magnitude=std::abs(value);
                return .5*(double(std::nextafter(magnitude,std::numeric_limits<float>::infinity()))-magnitude);
            };
            if(!fused) rounding[i]=up(rounding[i]+std::max(halfUlp(termLo),halfUlp(termHi)));
            const float fusedLo=std::fma(a,x,lo), fusedHi=std::fma(a,y,hi);
            lo=fused ? fusedLo : std::min(lo+termLo,fusedLo);
            hi=fused ? fusedHi : std::max(hi+termHi,fusedHi);
            rounding[i]=up(rounding[i]+std::max(halfUlp(lo),halfUlp(hi)));
        }
        if (!std::isfinite(lo)||!std::isfinite(hi)||lo>hi)
            throw std::runtime_error("invalid interval vertex");
        out.lo[i]=lo;
        out.hi[i]=hi;
    }
    return out;
}

struct Triangle {
    Face ids;
    Range bounds;
    std::array<Vec,3> edges;
    Vec normal;
};
struct Node {
    Range bounds;
    uint32_t begin,end;
    uint32_t left=0,right=0;
};
struct Surface {
    const std::vector<Vertex>& source;
    const Coordinates& lower;
    const Coordinates& upper;
    std::vector<Range> vertices;
    std::vector<Vec> rounding;
    std::vector<Triangle> triangles;
    std::vector<int> order;
    std::vector<Node> tree;

    uint32_t build(uint32_t begin,uint32_t end) {
        const auto index=uint32_t(tree.size());
        Range bounds=triangles[order[begin]].bounds;
        for(auto i=begin+1;i<end;++i) bounds.include(triangles[order[i]].bounds);
        tree.push_back({bounds,begin,end});
        if(end-begin>12) {
            const auto span=sub(bounds.hi,bounds.lo);
            const auto dim=std::max_element(span.begin(),span.end())-span.begin();
            const auto mid=begin+(end-begin)/2;
            std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,[&](int a,int b){
                const auto& x=triangles[a].bounds; const auto& y=triangles[b].bounds;
                return x.lo[dim]+x.hi[dim]<y.lo[dim]+y.hi[dim];
            });
            const auto l=build(begin,mid), r=build(mid,end);
            tree[index].left=l;
            tree[index].right=r;
        }
        return index;
    }
    Range relative(uint32_t a,uint32_t b) const {
        auto both=difference(vertices[a],vertices[b]);
        if(lower==upper) return both;
        const auto& x=source[a];
        const auto& y=source[b];
        for(int dim=0;dim<3;++dim) {
            // Common rigid-like movement cancels before interval multiplication.
            const double base=double(x[0][dim])-double(y[0][dim]);
            Interval affine{down(base),up(base)};
            for(int j=0;j<7;++j) {
                const double coefficient=double(x[j+1][dim])-double(y[j+1][dim]);
                affine=add(affine,mul({down(coefficient),up(coefficient)},{lower[j],upper[j]}));
            }
            const auto error=up(rounding[a][dim]+rounding[b][dim]);
            both.lo[dim]=std::max(both.lo[dim],down(affine.lo-error));
            both.hi[dim]=std::min(both.hi[dim],up(affine.hi+error));
            if(both.lo[dim]>both.hi[dim])
                throw std::runtime_error("disjoint absolute and relative bounds");
        }
        return both;
    }
    static bool separatesRelative(const std::array<Range,9>& rel,const Vec& axis) {
        bool pos=true,neg=true;
        for(const auto& delta:rel) {
            auto z=dot(delta,axis);
            pos=pos && z.lo>0;
            neg=neg && z.hi<0;
            if(!pos&&!neg) return false;
        }
        return pos||neg;
    }
    bool separates(const Triangle& a,const Triangle& b,const Vec& axis) const {
        if (dot(axis,axis)==0) return false;
        Interval p=dot(vertices[a.ids[0]],axis),q=dot(vertices[b.ids[0]],axis);
        for(int i=1;i<3;++i) {
            const auto x=dot(vertices[a.ids[i]],axis), y=dot(vertices[b.ids[i]],axis);
            p={std::min(p.lo,x.lo),std::max(p.hi,x.hi)};
            q={std::min(q.lo,y.lo),std::max(q.hi,y.hi)};
        }
        return p.hi<q.lo || q.hi<p.lo;
    }
    static bool coneSeparation(const std::array<Range,4>& v, const Vec& axis) {
        return std::all_of(v.begin(),v.end(),[&](const Range& a){return dot(a,axis).lo>0;});
    }
    static bool conesDisjoint(const std::array<Range,4>& v) {
        std::array<Vec,4> d;
        Vec average{};
        for(int i=0;i<4;++i) {
            const auto m=v[i].center();
            const double l=std::sqrt(dot(m,m));
            if(l==0) return false;
            d[i]=scaled(m,1/l);
            average=add(average,d[i]);
        }
        if(coneSeparation(v,average)) return true;
        // Candidate axes from the closest point on each convex-hull feature.
        // An axis is admitted only after its strict interval inequality.
        for(int i=0;i<4;++i) {
            if(coneSeparation(v,d[i])) return true;
            for(int j=i+1;j<4;++j) {
                if(coneSeparation(v,add(d[i],d[j]))) return true;
                for(int k=j+1;k<4;++k) {
                    const auto n=cross(sub(d[j],d[i]),sub(d[k],d[i]));
                    if(coneSeparation(v,scaled(n,dot(n,d[i])))) return true;
                }
            }
        }
        return false;
    }
    unsigned certify(const Triangle& a,const Triangle& b) const {
        Face shared{},onlyA{},onlyB{};
        int count=0,lenA=0,lenB=0;
        for(auto i:a.ids) {
            if(std::find(b.ids.begin(),b.ids.end(),i)!=b.ids.end()) shared[count++]=i;
            else onlyA[lenA++]=i;
        }
        for(auto i:b.ids)
            if(std::find(a.ids.begin(),a.ids.end(),i)==a.ids.end()) onlyB[lenB++]=i;
        if(count==3) return 4;  // Caller must quotient only declared duplicates.
        if(count==2) {
            const auto e=relative(shared[1],shared[0]);
            const auto u=relative(onlyA[0],shared[0]);
            const auto v=relative(onlyB[0],shared[0]);
            const auto n=cross(e,u), m=cross(e,v);
            if(dot(n,m).hi<0) return 0;
            const auto signedVolume=dot(n,v);
            return (signedVolume.lo>0 || signedVolume.hi<0) ? 0 : 3;
        }
        if(count==1) {
            if(conesDisjoint({relative(onlyA[0],shared[0]),relative(onlyA[1],shared[0]),
                              relative(shared[0],onlyB[0]),relative(shared[0],onlyB[1])}))
                return 0;
            return 2;
        }
        if(separates(a,b,a.normal)||separates(a,b,b.normal)) return 0;
        for(const auto& u:a.edges) {
            for(const auto& v:b.edges)
                if(separates(a,b,cross(u,v))) return 0;
            if(separates(a,b,cross(a.normal,u))) return 0;
        }
        for(const auto& v:b.edges)
            if(separates(a,b,cross(b.normal,v))) return 0;
        if(lower!=upper) {
            std::array<Range,9> delta;
            for(int i=0;i<3;++i)
                for(int j=0;j<3;++j)
                    delta[3*i+j]=relative(a.ids[i],b.ids[j]);
            if(separatesRelative(delta,a.normal)||separatesRelative(delta,b.normal)) return 0;
            for(const auto& u:a.edges) {
                for(const auto& v:b.edges)
                    if(separatesRelative(delta,cross(u,v))) return 0;
                if(separatesRelative(delta,cross(a.normal,u))) return 0;
            }
            for(const auto& v:b.edges)
                if(separatesRelative(delta,cross(b.normal,v))) return 0;
        }
        return 1;
    }
};

template<typename T> static void take(std::istream& in,T& result) {
    in.read(reinterpret_cast<char*>(&result),sizeof(T));
    if(!in) throw std::runtime_error("incomplete input");
}
template<typename T> static std::vector<T> readMany(std::istream& in,uint32_t count) {
    std::vector<T> result(count);
    in.read(reinterpret_cast<char*>(result.data()),sizeof(T)*count);
    if(!in) throw std::runtime_error("incomplete input array");
    return result;
}

int main(int argc,char** argv) try {
    if(argc<3||argc>4||(argc==4&&std::string_view(argv[3])!="fused"))
        throw std::runtime_error("usage: probe input.bin witness.csv [fused]");
    const bool fused=argc==4;
    std::ifstream file(argv[1],std::ios::binary);
    uint32_t nv{},nf{},windows{};
    take(file,nv);take(file,nf);take(file,windows);
    if(!nv||!nf||!windows||nv>2000000||nf>4000000)
        throw std::runtime_error("invalid geometry size");
    const auto vertices=readMany<Vertex>(file,nv);
    const auto faces=readMany<Face>(file,nf);
    std::ofstream witnesses(argv[2]);
    if(!witnesses) throw std::runtime_error("cannot write witness file");
    witnesses<<"window,reason,first_face,second_face\n";
    for(uint32_t w=0;w<windows;++w) {
        Coordinates lower{},upper{};
        take(file,lower);take(file,upper);
        for(int i=0;i<7;++i)
            if(!std::isfinite(lower[i])||!std::isfinite(upper[i])||lower[i]>upper[i])
                throw std::runtime_error("invalid coordinate range");
        Surface s{vertices,lower,upper,{},{},{},{},{}};
        s.vertices.reserve(nv);
        s.rounding.resize(nv);
        for(uint32_t i=0;i<nv;++i) s.vertices.push_back(pose(vertices[i],lower,upper,s.rounding[i],fused));
        std::array<uint64_t,5> fails{};
        s.triangles.reserve(nf);
        for(const auto& f:faces) {
            const auto& a=s.vertices.at(f[0]); const auto& b=s.vertices.at(f[1]); const auto& c=s.vertices.at(f[2]);
            const auto e0=sub(b.center(),a.center()), e1=sub(c.center(),b.center()),e2=sub(a.center(),c.center());
            const auto n=cross(e0,sub(c.center(),a.center()));
            if(dot(cross(s.relative(f[1],f[0]),s.relative(f[2],f[0])),n).lo<=0) {
                if(fails[0]<32) witnesses<<w<<",face,"<<s.triangles.size()<<",-1\n";
                ++fails[0];
            }
            auto bounds=a;
            bounds.include(b);
            bounds.include(c);
            s.triangles.push_back({f,bounds,{e0,e1,e2},n});
        }
        s.order.resize(nf);
        std::iota(s.order.begin(),s.order.end(),0);
        s.build(0,nf);
        uint64_t candidatePairs=0;
        for(uint32_t i=0;i<nf;++i) {
            const auto& a=s.triangles[i];
            std::vector<uint32_t> pending{0};
            while(!pending.empty()) {
                auto b=pending.back();
                pending.pop_back();
                const auto& leaf=s.tree[b];
                if(!a.bounds.overlaps(leaf.bounds)) continue;
                if(leaf.left) {
                    pending.push_back(leaf.left);
                    pending.push_back(leaf.right);
                    continue;
                }
                for(auto j=leaf.begin;j<leaf.end;++j) {
                    const auto k=uint32_t(s.order[j]);
                    const auto& other=s.triangles[k];
                    if(k<=i||!a.bounds.overlaps(other.bounds)) continue;
                    ++candidatePairs;
                    const auto status=s.certify(a,other);
                    if(status) {
                        if(fails[status]<32) witnesses<<w<<','<<status<<','<<i<<','<<k<<'\n';
                        ++fails[status];
                    }
                }
            }
        }
        std::cout<<w<<' '<<candidatePairs;
        for(auto count:fails) std::cout<<' '<<count;
        std::cout<<std::endl;
    }
    if(file.peek()!=EOF) throw std::runtime_error("unexpected input trailer");
    return 0;
} catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 1;
}
