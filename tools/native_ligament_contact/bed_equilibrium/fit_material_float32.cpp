// Local quantization of a qualified deformed material pose, with unchanged TetGen connectivity.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using V=std::array<double,3>;
using F=std::array<float,3>;
using T=std::array<int32_t,4>;
V sub(V a,V b){for(int i=0;i<3;i++)a[i]-=b[i];return a;}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V asDouble(F f){return {f[0],f[1],f[2]};}
template<class E> std::vector<E> read(const char* file){
 std::ifstream f(file,std::ios::binary|std::ios::ate);auto n=f.tellg();
 if(n<0 || size_t(n)%sizeof(E))throw std::runtime_error("incomplete input");
 std::vector<E> r(size_t(n)/sizeof(E));f.seekg(0);f.read((char*)r.data(),n);
 if(!f)throw std::runtime_error("input read error");return r;
}
struct M {double det,shape,penalty; bool valid;};
M calc(const std::vector<F>& y,const T& t,double ref){
 V a=asDouble(y[t[0]]),b=sub(asDouble(y[t[1]]),a),c=sub(asDouble(y[t[2]]),a),d=sub(asDouble(y[t[3]]),a);
 V bc=sub(b,c),bd=sub(b,d),cd=sub(c,d);
 double l=std::max({dot(b,b),dot(c,c),dot(d,d),dot(bc,bc),dot(bd,bd),dot(cd,cd)});
 double det=dot(b,cross(c,d)),j=det/ref,s=det/(l*std::sqrt(l));
 double p=std::max({0.,.4-j,j-2.5,1.-s/1e-5,(6e-18-det)/ref});
 bool valid=p==0 && det>6e-18;
 return {det,s,valid?0.:.2+std::log1p(p)+(det<=0?.15:0.),valid};
}
int main(int argc,char** argv){
 if(argc<5||argc>7)throw std::runtime_error("expected posed doubles, int32 tets, reference determinants and output");
 auto target=read<V>(argv[1]);auto t=read<T>(argv[2]);auto ref=read<double>(argv[3]);
 if(t.size()!=ref.size())throw std::runtime_error("cell count mismatch");
 std::vector<F> y; for(auto a:target){F f{float(a[0]),float(a[1]),float(a[2])}; y.push_back(f);}
 if(argc>=6&&std::string(argv[5])!="-"){auto warm=read<F>(argv[5]);if(warm.size()!=y.size())throw std::runtime_error("warm pose length");
  for(size_t i=0;i<warm.size();++i){V e=sub(asDouble(warm[i]),target[i]);if(!std::isfinite(dot(e,e))||dot(e,e)>4e-12)throw std::runtime_error("warm construction outside bound");}y=std::move(warm);}
 std::vector<uint8_t> fixed;
 if(argc==7){fixed=read<uint8_t>(argv[6]);if(fixed.size()!=y.size())throw std::runtime_error("fixed row count");}
 std::vector<std::vector<unsigned>> incid(y.size());
 for(unsigned c=0;c<t.size();c++){if(!(ref[c]>6e-18))throw std::runtime_error("invalid material reference");
  for(int i:t[c]){if(i<0||size_t(i)>=y.size())throw std::runtime_error("invalid node");incid[i].push_back(c);}}
 std::vector<double> cost(t.size());
 for(int iteration=0;iteration<21;++iteration){
  std::vector<unsigned> bad;size_t inv=0;
  for(unsigned c=0;c<t.size();c++){auto k=calc(y,t[c],ref[c]);cost[c]=k.penalty;if(!k.valid)bad.push_back(c);inv+=k.det<=0;}
  std::cout<<"iteration "<<iteration<<" failed "<<bad.size()<<" inverted "<<inv<<std::endl;
  if(bad.empty())break;
  std::stable_sort(bad.begin(),bad.end(),[&](unsigned a,unsigned b){return cost[a]>cost[b];});
  unsigned accepted=0;
  for(unsigned c:bad){
   if(cost[c]==0)continue;
   const auto& tet=t[c];auto state=calc(y,tet,ref[c]);
   double bestGain=0;int bestNode=-1;F bestF{};
   for(int slot=0;slot<4;slot++){
    int n=tet[slot];if(!fixed.empty()&&fixed[n])continue;F old=y[n];V now=asDouble(old),ulp{};
    V grad=cross(sub(asDouble(y[tet[(slot+2)%4]]),asDouble(y[tet[(slot+1)%4]])),
                 sub(asDouble(y[tet[(slot+3)%4]]),asDouble(y[tet[(slot+1)%4]])));
    if(slot%2==0)for(auto& v:grad)v=-v;
    for(int k=0;k<3;k++)ulp[k]=double(std::nextafter(old[k],std::numeric_limits<float>::infinity()))-old[k];
    int main=0;for(int k=1;k<3;k++)if(std::abs(grad[k]*ulp[k])>std::abs(grad[main]*ulp[main]))main=k;
    const int strongest=main;const bool few=bad.size()<=24;
    for(int axis=0;axis<(few?3:1);++axis){main=(strongest+axis)%3;
    if(grad[main]==0)continue;int i=(main+1)%3,j=(main+2)%3;
    const auto& star=incid[n];double before=0;for(auto row:star)before+=cost[row];
    auto tryPoint=[&](F q){
     if(q==old||(!fixed.empty()&&q[2]<0))return;
     V dx=sub(asDouble(q),target[n]);if(dot(dx,dx)>4e-12)return;
     y[n]=q;
     double after=0;
     for(auto row:star){after+=calc(y,t[row],ref[row]).penalty;if(after>before-bestGain+1e-12)break;}
     double gain=before-after;if(gain>bestGain+1e-10){bestGain=gain;bestNode=n;bestF=q;}
     y[n]=old;
    };
    // Account for cancellation in slivers: enumerate small lateral lattice moves
    // and solve the affine oriented-volume equation for the strongest axis.
    const int span=few?12:(iteration<4?2:4);
    for(int u=-span;u<=span;u++)for(int v=-span;v<=span;v++){
     F q=old;q[i]=float(now[i]+u*ulp[i]);q[j]=float(now[j]+v*ulp[j]);
     const double shapeFloor=state.det/state.shape*1.e-5/ref[c];
     for(double ratio:{.401,.60,1.,1.8,2.35,2.485,shapeFloor*1.001,shapeFloor*1.02}){
      if(ratio>2.5||ratio<.4||!std::isfinite(ratio))continue;
      const double z=(ratio*ref[c]-state.det-grad[i]*(double(q[i])-now[i])-grad[j]*(double(q[j])-now[j]))/grad[main];
      if(std::abs(z)>2.1e-6)continue;
      const float exact=float(now[main]+z);q[main]=exact;tryPoint(q);
      q[main]=std::nextafter(exact,-std::numeric_limits<float>::infinity());tryPoint(q);
      q[main]=std::nextafter(exact,std::numeric_limits<float>::infinity());tryPoint(q);
     }
     for(int k=-1;k<=1;k++){q[main]=float(now[main]+k*ulp[main]);tryPoint(q);}
    }
    }
   }
   if(bestNode>=0){y[bestNode]=bestF;for(auto row:incid[bestNode])cost[row]=calc(y,t[row],ref[row]).penalty;accepted++;}
  }
  if(!accepted&&iteration>=4)break;
 }
 std::ofstream out(argv[4],std::ios::binary);out.write((const char*)y.data(),y.size()*sizeof(F));
 return out?0:2;
}
