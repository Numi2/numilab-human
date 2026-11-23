#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <vector>
using Integer=__int128_t;
struct Arc {int32_t to,back;Integer available;};
struct Edge {int32_t a,b;double lower,upper,initial;};
static Integer units(double kg){return static_cast<Integer>(std::round(std::ldexp(kg,120)));}
static double kg(Integer u){return std::ldexp(static_cast<double>(u),-120);}
int main(){
    int32_t hdr[2];if(std::fread(hdr,sizeof(hdr),1,stdin)!=1||hdr[0]<2||hdr[1]<1)return 1;
    const int n=hdr[0],m=hdr[1],s=n,t=n+1;
    std::vector<std::vector<Arc>> g(n+2);std::vector<int32_t> owners(m),refs(m),level(n+2),cursor(n+2);
    std::vector<Integer> base(m),balance(n);
    auto add=[&](int a,int b,Integer forward,Integer backward){int j=g[a].size(),k=g[b].size();g[a].push_back({b,k,forward});g[b].push_back({a,j,backward});return j;};
    for(int j=0;j<m;++j){
        Edge e;if(std::fread(&e,sizeof(e),1,stdin)!=1||e.a<0||e.b<0||e.a>=n||e.b>=n||!std::isfinite(e.lower+e.upper+e.initial)||e.lower<0||e.lower>e.upper)return 2;
        auto lower=units(e.lower),upper=units(e.upper),f=std::clamp(units(e.initial),lower,upper);
        base[j]=lower;owners[j]=e.a;refs[j]=add(e.a,e.b,upper-f,f-lower);balance[e.a]-=f;balance[e.b]+=f;
    }
    for(int j=0;j<n;++j){if(balance[j]>0)add(s,j,balance[j],0);else if(balance[j]<0)add(j,t,-balance[j],0);}
    auto bfs=[&](){std::fill(level.begin(),level.end(),-1);std::queue<int> q;q.push(s);level[s]=0;
        while(!q.empty()){int u=q.front();q.pop();for(auto const& a:g[u])if(a.available&&level[a.to]<0){level[a.to]=level[u]+1;q.push(a.to);}}return level[t]>=0;};
    auto send=[&](auto&& self,int u,Integer f)->Integer{if(u==t)return f;
        for(int& j=cursor[u];j<int(g[u].size());++j){auto& a=g[u][j];if(!a.available||level[a.to]!=level[u]+1)continue;
            auto r=self(self,a.to,std::min(f,a.available));if(r){a.available-=r;g[a.to][a.back].available+=r;return r;}}
        return 0;};
    while(bfs()){std::fill(cursor.begin(),cursor.end(),0);while(send(send,s,Integer(1)<<125)){}}
    Integer lack=0;for(auto& e:g[s])lack+=e.available;double rem=kg(lack);
    std::fwrite(&rem,sizeof(rem),1,stdout);
    for(int j=0;j<m;++j){auto& e=g[owners[j]][refs[j]];double v=kg(base[j]+g[e.to][e.back].available);std::fwrite(&v,sizeof(v),1,stdout);}
    return std::ferror(stdout)?3:0;
}
