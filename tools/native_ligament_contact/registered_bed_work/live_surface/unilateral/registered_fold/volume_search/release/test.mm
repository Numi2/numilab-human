#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>
using V=std::array<double,3>;
static double det(V a,V b,V c){return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);}
int main(int argc,const char** argv){
 @autoreleasepool{
  if(argc!=2)return 2;
  auto gpu=MTLCreateSystemDefaultDevice();NSError* err=nil;
  auto lib=[gpu newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
  auto pipe=[&](NSString* name){
    auto f=[lib newFunctionWithName:[@"numi_matter_metal::nm_fem_registered_release_" stringByAppendingString:name]];
    return f?[gpu newComputePipelineStateWithFunction:f error:&err]:nil;
  };
  auto reset=pipe(@"reset"),check=pipe(@"check"),finish=pipe(@"finish");
  if(!reset||!check||!finish){NSLog(@"%@",err);return 3;}
  constexpr unsigned count=512;
  auto alloc=[&](size_t n){return [gpu newBufferWithLength:n options:MTLResourceStorageModeShared];};
  auto nb=alloc(4*count*sizeof(NMFEMNodeStateGPU)),tb=alloc(count*sizeof(NMTetrahedronGPU)),bb=alloc(count*sizeof(nm_float4));
  auto rb=alloc(4),sb=alloc(sizeof(NMMatterStatusGPU));auto queue=[gpu newCommandQueue];
  auto* n=static_cast<NMFEMNodeStateGPU*>(nb.contents);auto* t=static_cast<NMTetrahedronGPU*>(tb.contents);
  auto* b=static_cast<nm_float4*>(bb.contents);auto* status=static_cast<NMMatterStatusGPU*>(sb.contents);
  NMMatterDispatchGPU d{};d.tetrahedronCount=count;
  double maximum=0;unsigned firstCase=0;
  for(unsigned pass=0;pass<9;++pass){
    unsigned expected=0xffffffffu;double expectedJ=0;
    for(unsigned i=0;i<count;++i){
      const unsigned k=4*i;const bool ref=i%2;const bool flip=i%3==0;
      t[i]={};t[i].nodes={k,k+1,k+2,k+3};t[i].identity={16,15,0,NM_OBJECT_ACTIVE};
      const auto pos=[&](unsigned j,double z){return nm_float4{float(.25+(j==1?.01:(j==3?.008:0))),float(-.75+(j==2?.004:(j==3?.002:0))),float(.003+(j==3?z:0)),1};};
      const double thickness=flip?-3e-6:4e-6;
      std::array<V,4> loaded{},end{};
      const double target=pass && i>=pass*7 && (i%11==pass || i%9==pass)?(pass%2?.246715:.349):.351;
      for(unsigned j=0;j<4;++j){
        auto& a=n[k+j];a={};a.restAndFixed=pos(j,thickness);a.restAndFixed.w=0;
        auto y=pos(j,thickness*1.41864);a.positionAndMass=pos(j,thickness*1.41864*target);
        a.referenceDisplacementAndMode={0,0,float(j==3?thickness*(1.41864*target-1):0),float(ref)};
        a.referenceDisplacementLow.z=ref&&j==3?1.e-14f:0.f;
        loaded[j]={y.x,y.y,y.z};end[j]={a.positionAndMass.x,a.positionAndMass.y,ref?double(a.restAndFixed.z)+a.referenceDisplacementAndMode.z+a.referenceDisplacementLow.z:double(a.positionAndMass.z)};
      }
      auto volume=[](std::array<V,4> p){
        for(unsigned j=1;j<4;++j)for(unsigned v=0;v<3;++v)p[j][v]-=p[0][v];
        return det(p[1],p[2],p[3]);
      };
      double base=volume(loaded),current=volume(end);double cutoff=.35*base;float h=float(cutoff);
      b[i]={h,float(cutoff-double(h)),.496524f,base>0?1.f:-1.f};
      double J=current/base;
      if(expected==0xffffffffu && !(J>.35)){expected=i;expectedJ=J;}
    }
    *status={};
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
    auto go=[&](unsigned num){[e dispatchThreads:MTLSizeMake(num,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e memoryBarrierWithScope:MTLBarrierScopeBuffers];};
    [e setComputePipelineState:reset];[e setBuffer:rb offset:0 atIndex:0];go(1);
    [e setComputePipelineState:check];[e setBytes:&d length:sizeof(d) atIndex:0];[e setBuffer:tb offset:0 atIndex:1];
    [e setBuffer:nb offset:0 atIndex:2];[e setBuffer:bb offset:0 atIndex:3];[e setBuffer:rb offset:0 atIndex:4];go(count);
    [e setComputePipelineState:finish];[e setBuffer:tb offset:0 atIndex:0];[e setBuffer:nb offset:0 atIndex:1];[e setBuffer:bb offset:0 atIndex:2];[e setBuffer:rb offset:0 atIndex:3];[e setBuffer:sb offset:0 atIndex:4];go(1);
    [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 4;
    unsigned actual=*static_cast<unsigned*>(rb.contents);
    if(actual!=expected || ((!status->code)!=(expected==0xffffffffu)))return 5;
    if(status->code){
      if(status->failingIndex!=expected||status->diagnostics.w!=-913.f)return 6;
      maximum=std::max(maximum,std::abs(double(status->diagnostics.x)-expectedJ));++firstCase;
    }
  }
  if(maximum>2.e-6)return 7;
  std::printf("{\"device\":\"%s\",\"comparisons\":%u,\"failure_batches\":%u,\"max_ratio_error\":%.10g}\n",gpu.name.UTF8String,9*count,firstCase,maximum);
 }
}
