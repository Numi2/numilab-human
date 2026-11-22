#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/shared.h"
#include <cstdio>
#include <cstring>
#include <vector>
struct TestRow {float x[4][3],u[4][3],v[4][3],mass[4],inv[3][3],fixed,reference;};
int main(int argc,const char** argv) {
  @autoreleasepool {
    if(argc!=4 && argc!=5)return 2;
    const bool release=argc==5 && !std::strcmp(argv[4],"release");
    const bool joint=argc==5 && !std::strcmp(argv[4],"shared");
    const bool privateNodes=argc==5 && (!std::strcmp(argv[4],"private") || joint || release);
    const bool active=privateNodes || (argc==5 && !std::strcmp(argv[4],"active"));
    auto path=[&](int n){return [NSString stringWithUTF8String:argv[n]];};
    auto dev=MTLCreateSystemDefaultDevice();NSError* error=nil;
    auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:path(1)] error:&error];
    auto fn=[lib newFunctionWithName:active ? @"numi_matter_metal::nm_fem_registered_volume_active" : @"numi_matter_metal::nm_fem_registered_volume_direction"];
    if(!fn){NSLog(@"Missing volume-direction kernel");return 3;}
    auto pipe=[dev newComputePipelineStateWithFunction:fn error:&error];
    auto input=[NSData dataWithContentsOfFile:path(2)];
    if(!pipe || !input.length || input.length%sizeof(TestRow)){NSLog(@"%@",error);return 3;}
    const auto* rows=static_cast<const TestRow*>(input.bytes);
    std::uint32_t count=std::uint32_t(input.length/sizeof(TestRow));
    std::vector<NMFEMNodeStateGPU> nodes(count*4);
    std::vector<NMTetrahedronGPU> tets(count);
    std::vector<nm_float4> dir(count*4);
    std::vector<std::uint32_t> indices(count);
    auto vec=[](const float a[3],float w){return nm_float4{a[0],a[1],a[2],w};};
    for(std::uint32_t c=0;c<count;++c){
      const auto& r=rows[c];auto& t=tets[c];auto base=4*c;indices[c]=c;
      t.nodes={base,base+1,base+2,base+3};t.identity.w=NM_OBJECT_ACTIVE;
      if(joint && c%2)t.nodes.x=base-4;
      if(r.reference)t.identity.w|=NM_TETRAHEDRON_REFERENCE_DISPLACEMENT_GRADIENT;
      t.inverseRestRow0=vec(r.inv[0],0);t.inverseRestRow1=vec(r.inv[1],0);t.inverseRestRow2=vec(r.inv[2],0);
      for(unsigned k=0;k<4;++k){
        auto& n=nodes[base+k];n.restAndFixed=vec(r.x[k],k==0?r.fixed:0);
        n.referenceDisplacementAndMode=vec(r.u[k],r.reference);
        n.positionAndMass={r.x[k][0]+r.u[k][0],r.x[k][1]+r.u[k][1],r.x[k][2]+r.u[k][2],r.mass[k]};
        n.velocityAndInverseMass.w=1/r.mass[k];dir[base+k]=vec(r.v[k],13);
      }
    }
    NMMatterDispatchGPU d{};d.environmentCount=d.objectCount=1;d.femNodeCount=count*4;
    d.tetrahedronCount=count;d.gravityAndTimestep.w=.002f;d.numericalLimits.z=.01f;
    NMMicrostepGPU micro{};NMContinuumObjectGPU obj{};obj.flags=NM_OBJECT_ACTIVE;obj.representation=NM_REPRESENTATION_FEM;
    NMMaterialGPU mat{};mat.validity.x=.35f;mat.validity.y=2.5f;
    NMSchedulerStateGPU sched{};sched.numerical.w=1;
    NMAdaptiveStateGPU adaptive{};adaptive.activeRepresentation=obj.representation;NMFGMRESStateGPU state{};
    auto buf=[&](const void* b,size_t n){return [dev newBufferWithBytes:b length:n options:MTLResourceStorageModeShared];};
    auto pos=buf(nodes.data(),nodes.size()*sizeof(nodes[0]));auto out=buf(dir.data(),dir.size()*sizeof(dir[0]));
    auto elems=buf(tets.data(),tets.size()*sizeof(tets[0]));auto index=buf(indices.data(),count*4);
    auto q=[dev newCommandQueue];auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipe];
    [e setBytes:&d length:sizeof(d) atIndex:0];[e setBytes:&micro length:sizeof(micro) atIndex:1];
    [e setBytes:&obj length:sizeof(obj) atIndex:2];[e setBytes:&mat length:sizeof(mat) atIndex:3];
    [e setBuffer:pos offset:0 atIndex:4];[e setBuffer:pos offset:0 atIndex:5];[e setBuffer:out offset:0 atIndex:6];
    [e setBuffer:elems offset:0 atIndex:7];[e setBytes:&sched length:sizeof(sched) atIndex:8];
    [e setBytes:&adaptive length:sizeof(adaptive) atIndex:9];[e setBytes:&state length:sizeof(state) atIndex:10];
    [e setBuffer:index offset:0 atIndex:11];[e setBytes:&count length:sizeof(count) atIndex:12];
    if(active){
      std::vector<unsigned> claims(count*4);
      if(privateNodes){
        for(const auto& tet:tets){++claims[tet.nodes.x];++claims[tet.nodes.y];++claims[tet.nodes.z];++claims[tet.nodes.w];}
      }else{
        for(unsigned i=0;i<count*4;++i)claims[i]=i/4;
      }
      auto ownership=buf(claims.data(),claims.size()*4);
      [e setBuffer:ownership offset:0 atIndex:13];
    }
    if(release){
      const unsigned mode=2;std::vector<nm_float4> floors(count,{0,0,float(.35*1.41864),1});
      auto bounds=buf(floors.data(),floors.size()*sizeof(nm_float4));
      [e setBuffer:bounds offset:0 atIndex:14];[e setBytes:&mode length:sizeof(mode) atIndex:15];
    }
    [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [e endEncoding];[cb commit];[cb waitUntilCompleted];if(cb.status!=MTLCommandBufferStatusCompleted)return 4;
    if(![[NSData dataWithBytes:out.contents length:out.length] writeToFile:path(3) atomically:YES])return 5;
    std::puts(dev.name.UTF8String);
  }
}
