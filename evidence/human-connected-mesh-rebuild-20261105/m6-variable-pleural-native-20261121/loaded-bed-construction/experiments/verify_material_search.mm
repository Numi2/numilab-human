#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static double contract(simd_double3x3 a,simd_double3x3 b){
    return simd_dot(a.columns[0],b.columns[0])+simd_dot(a.columns[1],b.columns[1])+simd_dot(a.columns[2],b.columns[2]);
}
static double energy(simd_double3x3 f){
    double j=simd_determinant(f);
    if(j<=0)return INFINITY;
    double l=std::log(j);
    return 6000.*(contract(f,f)-3.)-12000.*l+60000.*l*l;
}
int main(int argc,char** argv){@autoreleasepool{
    if(argc!=2)return 2;
    auto dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
    auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];check(lib!=nil,"library");
    auto kernel=[&](NSString* name){auto p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];check(p!=nil,"pipeline");return p;};
    auto potential=kernel(@"nm_pelvic_trial_potential"),choose=kernel(@"nm_pelvic_choose_material_step");
    auto queue=[dev newCommandQueue];
    auto make=[&](size_t n){auto b=[dev newBufferWithLength:n options:MTLResourceStorageModeShared];check(b!=nil,"buffer");std::memset(b.contents,0,n);return b;};
    constexpr unsigned split=1190, nodes=split+4, tiles=(nodes+127)/128;
    std::array<id<MTLBuffer>,9> b{make(16),make(8),make(32),make(96),make(8),make(16*nodes),make(16*nodes),make(16*nodes),make(2*8*tiles*4)};
    unsigned dims[4]={2,nodes,tiles,0};std::memcpy(b[0].contents,dims,sizeof(dims));
    auto step=(simd_float4*)b[6].contents;auto x=(simd_float4*)b[5].contents;
    auto r=(simd_float4*)b[7].contents;auto scale=make(8);auto active=(unsigned*)b[1].contents;
    std::array<simd_double3x3,2> f,d;
    for(unsigned s=0;s<2;++s){
        unsigned off=s*split;active[s]=1;((float*)b[4].contents)[s]=1;
        ((simd_uint4*)b[2].contents)[s]=simd_make_uint4(off,off+1,off+2,off+3);
        f[s]=matrix_identity_double3x3;
        for(unsigned j=0;j<3;++j){
            ((simd_float4*)b[3].contents)[3*s+j][j]=1;
            simd_float4 h=simd_make_float4(.012f*(j+1),-.003f*j,.003f,0);
            simd_float4 dir=simd_make_float4(j?-.01f:(s?1.8f:.06f),.04f*(j+1),.02f,0);
            x[off+1+j]=h;step[off+1+j]=dir;r[off+1+j]=dir*(s?26000.f:150000.f);
            for(unsigned k=0;k<3;++k){f[s].columns[j][k]+=h[k];d[s].columns[j][k]=dir[k];}
        }
    }
    auto run=[&](){
        auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];
        [e setComputePipelineState:potential];for(unsigned i=0;i<b.size();++i)[e setBuffer:b[i] offset:0 atIndex:i];
        [e dispatchThreadgroups:MTLSizeMake(2*tiles,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];[e setComputePipelineState:choose];[e setBytes:&tiles length:4 atIndex:0];
        [e setBuffer:b[1] offset:0 atIndex:1];[e setBuffer:b[8] offset:0 atIndex:2];[e setBuffer:scale offset:0 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
        [cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"GPU");
    };
    run();double maxRelative=0;
    for(unsigned s=0;s<2;++s){
        double slope=0;
        for(unsigned j=1;j<4;++j){auto rr=r[s*split+j],dd=step[s*split+j];for(unsigned k=0;k<3;++k)slope+=double(rr[k])*dd[k];}
        auto invT=simd_transpose(simd_inverse(f[s]));
        double pd=12000.*contract(f[s],d[s])+(120000.*std::log(simd_determinant(f[s]))-12000.)*contract(invT,d[s]);
        double best=0,alpha=0;
        for(unsigned k=0;k<8;++k){
            double a=1./(1u<<k);auto ft=f[s];for(unsigned j=0;j<3;++j)ft.columns[j]-=a*d[s].columns[j];
            double ref=energy(ft)-energy(f[s])+a*(pd-slope);
            double seen=0;for(unsigned t=0;t<tiles;++t)seen+=((float*)b[8].contents)[8*(s*tiles+t)+k];
            if(!std::isfinite(ref)){check(std::isinf(seen),"inversion");continue;}
            double delta=std::abs(seen-ref)/std::max(1.,std::abs(ref));maxRelative=std::max(maxRelative,delta);
            check(delta<3e-5,"finite potential");
            if(ref<best){best=ref;alpha=a;}
        }
        check(((float*)scale.contents)[s]==alpha&&alpha>0,"bounded minimizer");
    }
    active[1]=0;((float*)scale.contents)[1]=.375f;
    std::memset(r,0,b[7].length);run();
    check(((float*)scale.contents)[0]==0,"zero descent");
    check(((float*)scale.contents)[1]==.375f,"inactive side");
    std::cout<<"side_count,candidate_count,max_scaled_error,inversion,zero_descent,inactive\n2,8,"<<maxRelative<<",pass,pass,pass\n";return 0;
}}
