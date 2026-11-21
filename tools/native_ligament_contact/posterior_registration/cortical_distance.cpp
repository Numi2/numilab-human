// Closest cortical feature on a closed, outward-oriented local mesh. No sampling grid.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>
#include <map>
using V=std::array<double,3>;
using T=std::array<int,3>;
static V sub(V a,V b){for(int i=0;i<3;++i)a[i]-=b[i];return a;}
static V add(V a,V b){for(int i=0;i<3;++i)a[i]+=b[i];return a;}
static V mul(V a,double s){for(auto& p:a)p*=s;return a;}
static double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static V unit(V a){return mul(a,1/std::sqrt(dot(a,a)));}
struct Box {V lo{INFINITY,INFINITY,INFINITY},hi{-INFINITY,-INFINITY,-INFINITY};
    void include(V p){for(int a=0;a<3;++a){lo[a]=std::min(lo[a],p[a]);hi[a]=std::max(hi[a],p[a]);}}
    double distance2(V p)const {V d{};for(int a=0;a<3;++a)d[a]=std::max({lo[a]-p[a],p[a]-hi[a],0.});return dot(d,d);}
};
struct Node {Box box;int first=0,end=0,left=-1,right=-1;};
struct Match {double squared=INFINITY;int face=-1;V point{},normal{};};
struct Surface {
    std::vector<V> x,n,c,vertex;std::vector<T> t;std::vector<std::array<V,3>> edge;
    std::vector<int> indices;std::vector<Node> tree;
    Surface(const double* v,int nv,const int* f,int nf):x(nv),n(nf),c(nf),vertex(nv),t(nf),edge(nf),indices(nf){
        std::map<std::pair<int,int>,V> shared;
        for(int i=0;i<nv;++i)std::copy_n(v+3*i,3,x[i].begin());
        for(int i=0;i<nf;++i){
            std::copy_n(f+3*i,3,t[i].begin());const auto a=x[t[i][0]],b=x[t[i][1]],d=x[t[i][2]];
            n[i]=unit(cross(sub(b,a),sub(d,a)));c[i]=mul(add(add(a,b),d),1./3.);
            for(int j=0;j<3;++j){int A=t[i][j],B=t[i][(j+1)%3],C=t[i][(j+2)%3];
                auto e=sub(x[B],x[A]),g=sub(x[C],x[A]);auto k=cross(e,g);
                double angle=std::atan2(std::sqrt(dot(k,k)),dot(e,g));
                vertex[A]=add(vertex[A],mul(n[i],angle));auto key=std::minmax(A,B);shared[key]=add(shared[key],n[i]);}
        }
        for(auto& p:vertex) p=unit(p);
        for(int i=0;i<nf;++i)for(int j=0;j<3;++j)edge[i][j]=unit(shared[std::minmax(t[i][j],t[i][(j+1)%3])]);
        std::iota(indices.begin(),indices.end(),0);build(0,nf);
    }
    int build(int first,int end){
        int index=static_cast<int>(tree.size());tree.emplace_back();Box box;
        for(int j=first;j<end;++j)for(int k:t[indices[j]])box.include(x[k]);
        tree[index].box=box;tree[index].first=first;tree[index].end=end;
        if(end-first<=8)return index;
        V length=sub(box.hi,box.lo);int axis=int(std::max_element(length.begin(),length.end())-length.begin()),mid=(first+end)/2;
        std::nth_element(indices.begin()+first,indices.begin()+mid,indices.begin()+end,[&](int a,int b){return c[a][axis]<c[b][axis];});
        int left=build(first,mid),right=build(mid,end);tree[index].left=left;tree[index].right=right;return index;
    }
    void sample(V p,int id,Match& best)const{
        auto a=x[t[id][0]],e=sub(x[t[id][1]],a),f=sub(x[t[id][2]],a),r=sub(p,a);
        auto k=cross(e,f);auto k2=dot(k,k);double u=dot(cross(r,f),k)/k2,v=dot(cross(e,r),k)/k2;
        auto test=[&](V q,V normal){auto d=sub(q,p);double sq=dot(d,d);if(sq<best.squared||(sq==best.squared&&id<best.face))best={sq,id,q,normal};};
        if(u>0&&v>0&&u+v<1){test(add(a,add(mul(e,u),mul(f,v))),n[id]);return;}
        for(int j=0;j<3;++j){int ia=t[id][j],ib=t[id][(j+1)%3];auto dir=sub(x[ib],x[ia]);
            double s=dot(sub(p,x[ia]),dir)/dot(dir,dir);
            if(s<=0)test(x[ia],vertex[ia]);else if(s>=1)test(x[ib],vertex[ib]);else test(add(x[ia],mul(dir,s)),edge[id][j]);}
    }
    void nearest(int j,V p,Match& best)const{
        const auto& b=tree[j];if(b.box.distance2(p)>best.squared)return;
        if(b.left<0){for(int i=b.first;i<b.end;++i)sample(p,indices[i],best);return;}
        int l=b.left,r=b.right;if(tree[l].box.distance2(p)>tree[r].box.distance2(p))std::swap(l,r);
        nearest(l,p,best);nearest(r,p,best);
    }
};
extern "C" void* cortex_create(const double* v,int nv,const int* f,int nf){return new Surface(v,nv,f,nf);}
extern "C" void cortex_destroy(void* s){delete static_cast<Surface*>(s);}
extern "C" void cortex_query(void* s,const double* pts,int count,double* results,int* ids){
    auto& mesh=*static_cast<Surface*>(s);
    for(int i=0;i<count;++i){V p{pts[i*3],pts[i*3+1],pts[i*3+2]};Match m;mesh.nearest(0,p,m);
        auto dx=sub(p,m.point);double dist=std::sqrt(m.squared);double sign=dot(dx,m.normal)<0?-1:1;
        results[7*i]=sign*dist;V normal=dist>1e-12?mul(dx,sign/dist):m.normal;
        for(int a=0;a<3;++a){results[7*i+1+a]=normal[a];results[7*i+4+a]=m.point[a];}ids[i]=m.face;}
}
