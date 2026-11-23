// A visibility graph in a section through both cortical entheses.
// Edges in the section remain admissible only if every open interval lies outside every bone.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

struct Point {double x,y;};
struct Segment {Point a,b; int bone;};
static Point sub(Point a,Point b){return {a.x-b.x,a.y-b.y};}
static double cross(Point a,Point b){return a.x*b.y-a.y*b.x;}
static double length(Point p){return std::hypot(p.x,p.y);}
static double dot(Point a,Point b){return a.x*b.x+a.y*b.y;}
constexpr double eps=1e-7; // millimetres, for double-precision section coincidences

static bool exterior(Point p,const std::vector<Segment>& walls){
    std::array<bool,186> inside{},boundary{};
    for(const auto& s:walls){
        const auto v=sub(s.b,s.a), w=sub(p,s.a);
        if(std::abs(cross(v,w))<=eps*length(v) &&
           dot(w,sub(p,s.b))<=eps*eps)boundary[s.bone]=true;
        if((s.a.y>p.y)!=(s.b.y>p.y)){
            double x=s.a.x+(p.y-s.a.y)*(s.b.x-s.a.x)/(s.b.y-s.a.y);
            if(x>p.x)inside[s.bone]=!inside[s.bone];
        }
    }
    for(size_t i=0;i<inside.size();i++)if(inside[i]&&!boundary[i])return false;
    return true;
}

static bool visible(Point a,Point b,const std::vector<Segment>& walls){
    const auto v=sub(b,a);const double n=length(v);
    if(n<eps)return false;
    std::vector<double> cuts{0.,1.};
    for(const auto& s:walls){
        if(std::max(s.a.x,s.b.x)+eps<std::min(a.x,b.x)||std::min(s.a.x,s.b.x)-eps>std::max(a.x,b.x)||
           std::max(s.a.y,s.b.y)+eps<std::min(a.y,b.y)||std::min(s.a.y,s.b.y)-eps>std::max(a.y,b.y))continue;
        const auto w=sub(s.b,s.a),d=sub(s.a,a);const auto den=cross(v,w);
        if(std::abs(den)<=eps*(n+length(w))){
            if(std::abs(cross(d,v))<=eps*n){
                cuts.push_back(std::clamp(dot(d,v)/(n*n),0.,1.));
                cuts.push_back(std::clamp(dot(sub(s.b,a),v)/(n*n),0.,1.));
            }
            continue;
        }
        double t=cross(d,w)/den, u=cross(d,v)/den;
        if(t<=eps/n||t>=1.-eps/n||u< -eps/length(w)||u>1.+eps/length(w))continue;
        // Crossing the interior of a boundary facet enters a closed cortical solid on one side.
        if(u>eps/length(w)&&u<1.-eps/length(w))return false;
        cuts.push_back(t);
    }
    std::sort(cuts.begin(),cuts.end());
    for(size_t j=1;j<cuts.size();++j){
        if((cuts[j]-cuts[j-1])*n<=2.*eps)continue;
        double t=(cuts[j]+cuts[j-1])*.5;
        if(!exterior({a.x+t*v.x,a.y+t*v.y},walls))return false;
    }
    return true;
}

extern "C" int section_path(const double* points,int n,const double* edges,const int* owners,
                             int m,double limit,int* route){
    std::vector<Point> p(n);
    for(int j=0;j<n;j++)p[j]={points[2*j],points[2*j+1]};
    std::vector<Segment> s;
    for(int j=0;j<m;j++){
        if(owners[j]<1||owners[j]>185)return -1;
        Point a{edges[4*j],edges[4*j+1]},b{edges[4*j+2],edges[4*j+3]};
        if(length(sub(a,b))>eps)s.push_back({a,b,owners[j]});
    }
    if(n<2)return -1;
    using Item=std::pair<double,int>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> open;
    std::vector<int> parent(n,-1);std::vector<double> d(n,std::numeric_limits<double>::infinity());
    std::vector<bool> closed(n,false);d[0]=0;open.emplace(length(sub(p[1],p[0])),0);
    while(!open.empty()){
        auto [est,i]=open.top();open.pop();
        if(est>limit)break;
        if(closed[i])continue;
        closed[i]=true;
        if(i==1){int size=0;for(int j=1;j!=-1;j=parent[j])route[size++]=j;
            std::reverse(route,route+size);return size;}
        for(int j=0;j<n;j++){
            if(closed[j])continue;
            const double cost=d[i]+length(sub(p[i],p[j]));
            const double key=cost+length(sub(p[j],p[1]));
            if(cost>=d[j]||key>limit||!visible(p[i],p[j],s))continue;
            parent[j]=i;d[j]=cost;open.emplace(key,j);
        }
    }
    return 0;
}
