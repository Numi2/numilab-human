#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <queue>
#include <vector>
struct Arc { int to,back; double balance; };
struct Edge { int32_t a,b; double cap; };
int main(){
    int32_t header[4];if(std::fread(header,sizeof(header),1,stdin)!=1)return 1;
    const int n=header[0],source=header[1],sink=header[2],count=header[3];
    if(n<2||source<0||sink<0||source>=n||sink>=n||count<0)return 2;
    std::vector<std::vector<Arc>> g(n);std::vector<int> level(n),it(n);
    for(int i=0;i<count;++i){Edge e;if(std::fread(&e,sizeof(e),1,stdin)!=1||e.a>=n||e.b>=n||e.a<0||e.b<0||!std::isfinite(e.cap)||e.cap<0)return 3;
        Arc a{e.b,int(g[e.b].size()),e.cap},b{e.a,int(g[e.a].size()),0.};g[e.a].push_back(a);g[e.b].push_back(b);}
    auto levelGraph=[&](){std::fill(level.begin(),level.end(),-1);std::queue<int> q;q.push(source);level[source]=0;
        while(!q.empty()){int u=q.front();q.pop();for(const auto& e:g[u])if(e.balance>1e-22&&level[e.to]<0){level[e.to]=level[u]+1;q.push(e.to);}}return level[sink]>=0;};
    auto push=[&](auto&& self,int u,double limit)->double{if(u==sink)return limit;
        for(int& j=it[u];j<int(g[u].size());++j){auto& e=g[u][j];if(e.balance<=1e-22||level[e.to]!=level[u]+1)continue;
            double val=self(self,e.to,std::min(limit,e.balance));if(val<=1e-22)continue;
            e.balance-=val;g[e.to][e.back].balance+=val;return val;}
        return 0.;};
    while(levelGraph()){std::fill(it.begin(),it.end(),0);while(push(push,source,std::numeric_limits<double>::max())>1e-22){}}
    for(const auto& e:g[source])if(e.balance>1e-17)std::printf("s %d %.17g\n",e.to,e.balance);
    for(const auto& e:g[sink]){const double d=g[e.to][e.back].balance;if(d>1e-17)std::printf("r %d %.17g\n",e.to,d);}
    for(int j=0;j<n;++j)if(level[j]>=0)std::printf("c %d\n",j);
}
