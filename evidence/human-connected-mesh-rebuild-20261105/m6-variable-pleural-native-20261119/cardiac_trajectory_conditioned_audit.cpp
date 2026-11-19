// Authoring test for the common-field cardiac deformation in native local metres.
// Certifies a seven-term affine field with an explicit two-float evaluation bound.
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
using PackedVertex = std::array<FVec, 8>;
using Vertex = std::array<Vec, 8>;
using Coordinates = std::array<float, 7>;

struct CoordinateTube {
    Coordinates first,last,residualLow,residualHigh;
    uint32_t pivot;
};

static std::pair<Vertex,Vec> mapTube(const PackedVertex& original,const CoordinateTube& tube) {
    Vertex v{};
    Vec reserve{};
    for(int dim=0;dim<3;++dim) {
        v[0][dim]=original[0][dim];
        double magnitude=std::abs(v[0][dim]);
        int column=2;
        for(int j=0;j<7;++j) {
            const double d=original[j+1][dim];
            const double a=tube.first[j], b=tube.last[j];
            v[0][dim]=std::fma(d,a,v[0][dim]);
            v[1][dim]=std::fma(d,b-a,v[1][dim]);
            if(uint32_t(j)!=tube.pivot) v[column++][dim]=d;
            magnitude+=std::abs(d)*(std::abs(a)+std::abs(b)+
                std::max(std::abs(double(tube.residualLow[j])),std::abs(double(tube.residualHigh[j]))));
        }
        // Covers the compensated GPU evaluation in the original coordinates
        // plus the bounded double arithmetic forming the two transformed rows.
        reserve[dim]=std::ldexp(magnitude,-40)+std::ldexp(magnitude,-47)+std::ldexp(1.,-122);
    }
    return {v,reserve};
}

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
enum class Precision { original, roundedOnce, recentered, split };
static Range accuratePose(const Vertex& v,const Coordinates& lower,const Coordinates& upper,
                          Vec& rounding,Precision precision,const Vec* tubeReserve=nullptr) {
    const Vec origin = precision==Precision::recentered ? Vec{.078125,0.,-.0234375} : Vec{};
    Range out;
    for(int dim=0;dim<3;++dim) {
        auto bound=Interval{down(double(v[0][dim])-origin[dim]),up(double(v[0][dim])-origin[dim])};
        for(int k=0;k<7;++k) {
            const double coefficient=v[k+1][dim];
            bound=add(bound,mul({coefficient,coefficient},{lower[k],upper[k]}));
        }
        if(precision==Precision::split) {
            // Two-float geometry evaluation with 40 effective significand
            // bits. It cannot be admitted before a matching GPU check.
            double maxAbs=std::max(std::abs(bound.lo),std::abs(bound.hi));
            double totalMagnitude=std::abs(double(v[0][dim]));
            for(int k=0;k<7;++k)
                totalMagnitude+=std::abs(double(v[k+1][dim]))*
                    std::max(std::abs(double(lower[k])),std::abs(double(upper[k])));
            const double boundError=up(tubeReserve?(*tubeReserve)[dim]:
                std::max(std::ldexp(std::max(maxAbs,totalMagnitude),-40),std::ldexp(1.,-122)));
            out.lo[dim]=down(bound.lo-boundError);
            out.hi[dim]=up(bound.hi+boundError);
            rounding[dim]=boundError;
        } else {
            const float low=static_cast<float>(bound.lo), high=static_cast<float>(bound.hi);
            out.lo[dim]=low;
            out.hi[dim]=high;
            const float magnitude=std::max(std::abs(low),std::abs(high));
            rounding[dim]=up(.5*(double(std::nextafter(magnitude,std::numeric_limits<float>::infinity()))-magnitude));
        }
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
    static Interval determinant(const Range& a,const Range& b,const Range& c) {
        return dot(cross(a,b),c);
    }
    Range coefficientDifference(uint32_t a,uint32_t b,int coefficient) const {
        Range result;
        for(int dim=0;dim<3;++dim) {
            const double x=double(source[a][coefficient][dim])-double(source[b][coefficient][dim]);
            result.lo[dim]=down(x);
            result.hi[dim]=up(x);
        }
        return result;
    }
    static Range addVector(const Range& a,const Range& b) {
        Range out;
        for(int i=0;i<3;++i) {
            const auto r=add(component(a,i),component(b,i));
            out.lo[i]=r.lo;
            out.hi[i]=r.hi;
        }
        return out;
    }
    static Range multiply(const Range& a,const Interval b) {
        Range out;
        for(int i=0;i<3;++i) {
            const auto r=mul(component(a,i),b);
            out.lo[i]=r.lo;
            out.hi[i]=r.hi;
        }
        return out;
    }
    using Edge=std::array<uint32_t,2>;
    Interval movingDeterminant(const std::array<Edge,3>& edges) const {
        std::array<Range,3> center,changes{};
        std::array<std::array<Range,7>,3> coefficients;
        std::array<double,7> radii;
        for(int j=0;j<7;++j) radii[j]=(double(upper[j])-double(lower[j]))*.5;
        for(int k=0;k<3;++k) {
            center[k]=coefficientDifference(edges[k][0],edges[k][1],0);
            for(int j=0;j<7;++j) {
                const double q=(double(upper[j])+double(lower[j]))*.5;
                coefficients[k][j]=coefficientDifference(edges[k][0],edges[k][1],j+1);
                center[k]=addVector(center[k],multiply(coefficients[k][j],{q,q}));
            }
            for(int dim=0;dim<3;++dim) {
                const double error=up(rounding[edges[k][0]][dim]+rounding[edges[k][1]][dim]);
                center[k].lo[dim]=down(center[k].lo[dim]-error);
                center[k].hi[dim]=up(center[k].hi[dim]+error);
            }
        }
        // Subtract constant multiples of preceding columns. A determinant
        // is unchanged by these operations. Centering the interval basis
        // before bounding the remainder avoids cancellation for thin faces.
        for(int dst=1;dst<3;++dst) {
            for(int src=0;src<dst;++src) {
                const auto u=center[src].center(),v=center[dst].center();
                const auto squared=dot(u,u);
                if(squared<=0) continue;
                const auto scale=-dot(u,v)/squared;
                if(!std::isfinite(scale)) continue;
                center[dst]=addVector(center[dst],multiply(center[src],{scale,scale}));
                for(int j=0;j<7;++j)
                    coefficients[dst][j]=addVector(coefficients[dst][j],
                        multiply(coefficients[src][j],{scale,scale}));
            }
        }
        for(int k=0;k<3;++k) for(int j=0;j<7;++j)
            changes[k]=addVector(changes[k],multiply(coefficients[k][j],{-radii[j],radii[j]}));
        auto volume=determinant(center[0],center[1],center[2]);
        for(int j=0;j<7;++j) {
            const auto slope=add(add(determinant(coefficients[0][j],center[1],center[2]),
                                     determinant(center[0],coefficients[1][j],center[2])),
                                     determinant(center[0],center[1],coefficients[2][j]));
            volume=add(volume,mul(slope,{-radii[j],radii[j]}));
        }
        for(int k=0;k<3;++k) {
            auto term=changes;
            term[k]=center[k];
            volume=add(volume,determinant(term[0],term[1],term[2]));
        }
        volume=add(volume,determinant(changes[0],changes[1],changes[2]));
        return volume;
    }
    bool edgeVolumeStrict(uint32_t origin,std::array<uint32_t,3> endpoints) const {
        const auto value=movingDeterminant({Edge{endpoints[0],origin},Edge{endpoints[1],origin},
                                            Edge{endpoints[2],origin}});
        return value.lo>0 || value.hi<0;
    }
    bool positiveArea(const Face& face,const Vec& reference) const {
        const Edge a{face[1],face[0]},b{face[2],face[0]};
        if(dot(cross(relative(a[0],a[1]),relative(b[0],b[1])),reference).lo>0)
            return true;
        if(lower==upper) return false;

        std::array<Edge,2> edges{a,b};
        std::array<Range,2> center{},remainder{};
        std::array<std::array<Range,7>,2> coefficient;
        std::array<double,7> radii;
        for(int j=0;j<7;++j) radii[j]=(double(upper[j])-double(lower[j]))*.5;
        for(int k=0;k<2;++k) {
            center[k]=coefficientDifference(edges[k][0],edges[k][1],0);
            for(int j=0;j<7;++j) {
                const double q=(double(upper[j])+double(lower[j]))*.5;
                coefficient[k][j]=coefficientDifference(edges[k][0],edges[k][1],j+1);
                center[k]=addVector(center[k],multiply(coefficient[k][j],{q,q}));
                remainder[k]=addVector(remainder[k],multiply(coefficient[k][j],{-radii[j],radii[j]}));
            }
            for(int d=0;d<3;++d) {
                const auto error=up(rounding[edges[k][0]][d]+rounding[edges[k][1]][d]);
                center[k].lo[d]=down(center[k].lo[d]-error);
                center[k].hi[d]=up(center[k].hi[d]+error);
            }
        }
        auto result=dot(cross(center[0],center[1]),reference);
        for(int j=0;j<7;++j) {
            auto derivative=add(dot(cross(coefficient[0][j],center[1]),reference),
                                dot(cross(center[0],coefficient[1][j]),reference));
            result=add(result,mul(derivative,{-radii[j],radii[j]}));
        }
        return add(result,dot(cross(remainder[0],remainder[1]),reference)).lo>0;
    }
    bool movingFaceSeparation(const Triangle& a,const Triangle& b) const {
        const auto origin=a.ids[0];
        bool pos=true,neg=true;
        for(auto id:b.ids) {
            const auto value=movingDeterminant({Edge{a.ids[1],origin},Edge{a.ids[2],origin},Edge{id,origin}});
            pos=pos&&value.lo>0;
            neg=neg&&value.hi<0;
            if(!pos&&!neg) return false;
        }
        return pos||neg;
    }
    bool movingEdgeSeparation(const Triangle& a,const Triangle& b) const {
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
            const Edge e{a.ids[(i+1)%3],a.ids[i]}, f{b.ids[(j+1)%3],b.ids[j]};
            bool pos=true,neg=true;
            for(auto v:{a.ids[i],a.ids[(i+2)%3]})
                for(auto w:{b.ids[j],b.ids[(j+2)%3]}) {
                    const auto value=movingDeterminant({e,f,Edge{v,w}});
                    pos=pos&&value.lo>0;
                    neg=neg&&value.hi<0;
                }
            if(pos||neg) return true;
        }
        return false;
    }
    Interval project(uint32_t a,uint32_t b,const Vec& axis) const {
        auto bound=dot(coefficientDifference(a,b,0),axis);
        for(int k=0;k<7;++k) {
            auto derivative=dot(coefficientDifference(a,b,k+1),axis);
            bound=add(bound,mul(derivative,{lower[k],upper[k]}));
        }
        Interval error{};
        for(int j=0;j<3;++j) {
            const auto e=up(rounding[a][j]+rounding[b][j]);
            error=add(error,mul({e,e},{std::abs(axis[j]),std::abs(axis[j])}));
        }
        return {down(bound.lo-error.hi),up(bound.hi+error.hi)};
    }
    bool separatesProjected(const Triangle& a,const Triangle& b,const Vec& axis) const {
        bool pos=true,neg=true;
        for(auto v:a.ids) for(auto w:b.ids) {
            const auto bound=project(v,w,axis);
            pos=pos && bound.lo>0;
            neg=neg && bound.hi<0;
            if(!pos&&!neg) return false;
        }
        return pos||neg;
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
    bool conesDisjoint(const std::array<Edge,4>& edges) const {
        std::array<Range,4> v;
        for(int i=0;i<4;++i) v[i]=relative(edges[i][0],edges[i][1]);
        const auto valid=[&](const Vec& axis){
            if(coneSeparation(v,axis)) return true;
            return lower!=upper&&std::all_of(edges.begin(),edges.end(),[&](const Edge& edge){
                return project(edge[0],edge[1],axis).lo>0;
            });
        };
        std::array<Vec,4> d;
        Vec average{};
        for(int i=0;i<4;++i) {
            const auto m=v[i].center();
            const double l=std::sqrt(dot(m,m));
            if(l==0) return false;
            d[i]=scaled(m,1/l);
            average=add(average,d[i]);
        }
        if(valid(average)) return true;
        // Candidate axes from the closest point on each convex-hull feature.
        // An axis is admitted only after its strict interval inequality.
        for(int i=0;i<4;++i) {
            if(valid(d[i])) return true;
            for(int j=i+1;j<4;++j) {
                if(valid(add(d[i],d[j]))) return true;
                for(int k=j+1;k<4;++k) {
                    const auto n=cross(sub(d[j],d[i]),sub(d[k],d[i]));
                    if(valid(scaled(n,dot(n,d[i])))) return true;
                }
            }
        }
        if(lower!=upper) {
            // The alternating three-column minors form the null vector of
            // these four signed rays. Opposite strict cofactor signs prove
            // rank three and exclude a nonnegative null vector everywhere
            // in the interval. Gordan's alternative then gives a strict
            // separating plane, even if no fixed plane works throughout.
            bool positive=false, negative=false;
            for(int i=0;i<4;++i) {
                std::array<Edge,3> triple;
                for(int j=0,k=0;j<4;++j) if(j!=i) triple[k++]=edges[j];
                const auto minor=movingDeterminant(triple);
                positive|=i%2 ? minor.hi<0 : minor.lo>0;
                negative|=i%2 ? minor.lo>0 : minor.hi<0;
                if(positive&&negative) return true;
            }
            // A separator can follow the local geometry. Try normals defined
            // by each adjacent pair of cone rays, with a small inward bias.
            // This avoids independent interval error on almost coplanar rays.
            constexpr double blend=.5;
            for(int i=0;i<4;++i) for(int j=i+1;j<4;++j) {
                auto axis=cross(d[i],d[j]);
                const auto positive=dot(axis,average)>=0;
                const double sign=positive?1.:-1.;
                const auto unitBias=scaled(add(d[i],d[j]),blend);
                const auto pointCheck=[&](int k,double bias){
                    return sign*dot(v[k].center(),axis)+bias*dot(v[k].center(),unitBias)>0;
                };
                for(double bias:{1.e-3,1.e-5,1.e-7,1.e-9}) {
                    bool separated=true;
                    if(!(pointCheck(0,bias)&&pointCheck(1,bias)&&
                         pointCheck(2,bias)&&pointCheck(3,bias))) continue;
                    for(int k=0;k<4 && separated;++k) {
                        auto bound=project(edges[k][0],edges[k][1],scaled(unitBias,bias));
                        if(k!=i&&k!=j) {
                            const auto determinant=movingDeterminant({edges[i],edges[j],edges[k]});
                            const auto lengthProduct=std::sqrt(dot(v[i].center(),v[i].center()))*
                                std::sqrt(dot(v[j].center(),v[j].center()));
                            const double normalization=sign/lengthProduct;
                            bound=add(bound,mul(determinant,{normalization,normalization}));
                        }
                        separated=bound.lo>0;
                    }
                    if(separated) return true;
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
            if(lower!=upper) {
                // Overlap beyond the common edge requires both face normals
                // (using the same directed edge) to point to the same side.
                // A common scalar projection with opposite signs rules it
                // out and preserves the correlated affine edge motion.
                const auto axis=sub(n.center(),m.center());
                if(positiveArea({shared[0],shared[1],onlyA[0]},axis)&&
                   positiveArea({shared[0],shared[1],onlyB[0]},scaled(axis,-1)))
                    return 0;
            }
            const auto signedVolume=dot(n,v);
            return (signedVolume.lo>0 || signedVolume.hi<0 ||
                    (lower!=upper&&edgeVolumeStrict(shared[0],{shared[1],onlyA[0],onlyB[0]}))) ? 0 : 3;
        }
        if(count==1) {
            if(conesDisjoint({Edge{onlyA[0],shared[0]},Edge{onlyA[1],shared[0]},
                              Edge{shared[0],onlyB[0]},Edge{shared[0],onlyB[1]}}))
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
            const auto separatesEither=[&](const Vec& axis){
                return separatesRelative(delta,axis)||separatesProjected(a,b,axis);
            };
            if(separatesEither(a.normal)||separatesEither(b.normal)) return 0;
            for(const auto& u:a.edges) {
                for(const auto& v:b.edges)
                    if(separatesEither(cross(u,v))) return 0;
                if(separatesEither(cross(a.normal,u))) return 0;
            }
            for(const auto& v:b.edges)
                if(separatesEither(cross(b.normal,v))) return 0;
            if(movingFaceSeparation(a,b)||movingFaceSeparation(b,a)||movingEdgeSeparation(a,b)) return 0;
        }
        return 1;
    }
};

static Triangle makeTriangle(const Surface& s,const Face& f) {
    const auto& a=s.vertices.at(f[0]);
    const auto& b=s.vertices.at(f[1]);
    const auto& c=s.vertices.at(f[2]);
    const auto e0=sub(b.center(),a.center());
    const auto e1=sub(c.center(),b.center());
    const auto e2=sub(a.center(),c.center());
    auto bounds=a;
    bounds.include(b);
    bounds.include(c);
    return {f,bounds,{e0,e1,e2},cross(e0,sub(c.center(),a.center()))};
}

struct PairRefinement {
    std::vector<Vertex> source;
    std::vector<Vec> error;
    std::array<Face,2> faces{};
    std::array<double,7> rate{};
    unsigned budget=255;
    unsigned deepest=0;

    PairRefinement(const Surface& all,const std::vector<Vec>& reserve,const Face& a,const Face& b) {
        std::vector<uint32_t> ids;
        for(unsigned f=0;f<2;++f) for(int i=0;i<3;++i) {
            auto id=(f?b:a)[i];
            const auto at=std::find(ids.begin(),ids.end(),id);
            faces[f][i]=uint32_t(at-ids.begin());
            if(at==ids.end()) {
                ids.push_back(id);
                source.push_back(all.source[id]);
                error.push_back(reserve[id]);
            }
        }
        for(int j=0;j<7;++j) for(int dim=0;dim<3;++dim) {
            double lo=source[0][j+1][dim],hi=lo;
            for(const auto& row:source) {
                lo=std::min(lo,row[j+1][dim]);
                hi=std::max(hi,row[j+1][dim]);
            }
            rate[j]=std::max(rate[j],hi-lo);
        }
    }
    bool prove(const Coordinates& lower,const Coordinates& upper,unsigned depth) {
        if(!budget) return false;
        --budget;
        deepest=std::max(deepest,depth);
        Surface s{source,lower,upper,{},{},{},{},{}};
        s.rounding.resize(source.size());
        s.vertices.reserve(source.size());
        for(size_t j=0;j<source.size();++j)
            s.vertices.push_back(accuratePose(source[j],lower,upper,s.rounding[j],Precision::split,&error[j]));
        const auto a=makeTriangle(s,faces[0]),b=makeTriangle(s,faces[1]);
        if(!a.bounds.overlaps(b.bounds)||!s.certify(a,b)) return true;
        if(depth>=24) return false;
        int pivot=-1;
        float midpoint=0.f;
        double variation=0;
        for(int j=0;j<7;++j) {
            const double width=double(upper[j])-double(lower[j]);
            const float next=float(double(lower[j])+width*.5);
            const double size=rate[j]*width;
            if(next>lower[j]&&next<upper[j]&&size>variation) {
                pivot=j;
                midpoint=next;
                variation=size;
            }
        }
        if(pivot<0) return false;
        auto lowerHalfUpper=upper,upperHalfLower=lower;
        lowerHalfUpper[pivot]=upperHalfLower[pivot]=midpoint;
        return prove(lower,lowerHalfUpper,depth+1)&&prove(upperHalfLower,upper,depth+1);
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
    if(argc!=4&&argc!=6) throw std::runtime_error("usage: probe input.bin witness.csv split-tube [first end]");
    const std::string_view requested(argv[3]);
    Precision precision=Precision::original;
    if(requested=="rounded-once") precision=Precision::roundedOnce;
    if(requested=="recentered") precision=Precision::recentered;
    if(requested=="split-tube") precision=Precision::split;
    if(precision==Precision::original) throw std::runtime_error("unknown precision policy");
    constexpr uint64_t witnessBudget=50000;
    std::ifstream file(argv[1],std::ios::binary);
    uint32_t nv{},nf{},windows{};
    take(file,nv);take(file,nf);take(file,windows);
    if(!nv||!nf||!windows||nv>2000000||nf>4000000)
        throw std::runtime_error("invalid geometry size");
    const auto packed=readMany<PackedVertex>(file,nv);
    const auto faces=readMany<Face>(file,nf);
    std::ofstream witnesses(argv[2]);
    if(!witnesses) throw std::runtime_error("cannot write witness file");
    witnesses<<"window,reason,first_face,second_face\n";
    const unsigned first=argc==6?std::stoul(argv[4]):0u;
    const unsigned end=argc==6?std::stoul(argv[5]):windows;
    if(first>=end||end>windows) throw std::runtime_error("invalid window subset");
    for(uint32_t w=0;w<windows;++w) {
        CoordinateTube tube;
        take(file,tube.first);take(file,tube.last);take(file,tube.residualLow);take(file,tube.residualHigh);
        take(file,tube.pivot);
        if(w<first||w>=end) continue;
        if(tube.pivot>=7||tube.first[tube.pivot]==tube.last[tube.pivot]||
           tube.residualLow[tube.pivot]!=0||tube.residualHigh[tube.pivot]!=0)
            throw std::runtime_error("invalid tube axis");
        Coordinates lower{},upper{};
        upper[0]=1;
        int k=1;
        for(int j=0;j<7;++j) {
            if(!std::isfinite(tube.first[j])||!std::isfinite(tube.last[j])||
               !std::isfinite(tube.residualLow[j])||!std::isfinite(tube.residualHigh[j])||
               tube.residualLow[j]>tube.residualHigh[j])
                throw std::runtime_error("invalid tube coordinate range");
            if(uint32_t(j)!=tube.pivot) {
                lower[k]=tube.residualLow[j];
                upper[k++]=tube.residualHigh[j];
            }
        }
        std::vector<Vertex> vertices(nv);
        std::vector<Vec> reserve(nv);
        for(uint32_t i=0;i<nv;++i) {
            auto mapped=mapTube(packed[i],tube);
            vertices[i]=mapped.first;
            reserve[i]=mapped.second;
        }
        Surface s{vertices,lower,upper,{},{},{},{},{}};
        s.vertices.reserve(nv);
        s.rounding.resize(nv);
        for(uint32_t i=0;i<nv;++i) s.vertices.push_back(accuratePose(vertices[i],lower,upper,s.rounding[i],precision,&reserve[i]));
        std::array<uint64_t,5> fails{};
        s.triangles.reserve(nf);
        for(const auto& f:faces) {
            const auto& a=s.vertices.at(f[0]); const auto& b=s.vertices.at(f[1]); const auto& c=s.vertices.at(f[2]);
            const auto e0=sub(b.center(),a.center()), e1=sub(c.center(),b.center()),e2=sub(a.center(),c.center());
            const auto n=cross(e0,sub(c.center(),a.center()));
            if(!s.positiveArea(f,n)) {
                if(fails[0]<witnessBudget) witnesses<<w<<",face,"<<s.triangles.size()<<",-1\n";
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
        uint64_t candidatePairs=0,refined=0,visits=0;
        for(uint32_t i=0;i<nf;++i) {
            const auto& a=s.triangles[i];
            std::array<uint32_t,64> pending{};
            std::size_t active=1;
            while(active) {
                auto b=pending[--active];
                const auto& leaf=s.tree[b];
                if(!a.bounds.overlaps(leaf.bounds)) continue;
                if(leaf.left) {
                    if(active+2>pending.size()) throw std::runtime_error("BVH stack width exceeded");
                    pending[active++]=leaf.left;
                    pending[active++]=leaf.right;
                    continue;
                }
                for(auto j=leaf.begin;j<leaf.end;++j) {
                    const auto k=uint32_t(s.order[j]);
                    const auto& other=s.triangles[k];
                    if(k<=i||!a.bounds.overlaps(other.bounds)) continue;
                    ++candidatePairs;
                    const auto status=s.certify(a,other);
                    if(status&&status!=4) {
                        PairRefinement candidate(s,reserve,a.ids,other.ids);
                        const bool ok=candidate.prove(lower,upper,0);
                        visits+=255u-candidate.budget;
                        refined+=ok;
                        if(ok) continue;
                    }
                    if(status) {
                        if(fails[status]<witnessBudget) witnesses<<w<<','<<status<<','<<i<<','<<k<<'\n';
                        ++fails[status];
                    }
                }
            }
        }
        witnesses.flush();
        std::cout<<w<<' '<<candidatePairs;
        for(auto count:fails) std::cout<<' '<<count;
        std::cout<<' '<<refined<<' '<<visits<<std::endl;
    }
    if(file.peek()!=EOF) throw std::runtime_error("unexpected input trailer");
    return 0;
} catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 1;
}
