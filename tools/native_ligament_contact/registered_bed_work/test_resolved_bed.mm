#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>
#include <array>
#include "metalrobo/engine_types.h"
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "metalrobo/numi_human_support_geometry_gpu.h"
#include "NumiHumanResolvedBed.hpp"

template<class T> id<MTLBuffer> buf(id<MTLDevice> dev,const std::vector<T>& v) {
    return [dev newBufferWithBytes:v.data() length:v.size()*sizeof(T) options:MTLResourceStorageModeShared];
}

static double stress(double s,double mu,double lambda) {return mu*(1/s-s)-lambda*std::log(s)/s;}
static double exactPressure(double d,double hs,double hf) {
    if(d<=0) return 0.0;
    // Independent split of the measured shortening, equal normal traction.
    double a=std::max(0.0,d-hf),b=std::min(d,hs);
    for(int i=0;i<70;++i) {
        double u=(a+b)/2;
        if(stress(1-u/hs,13255.0336,658333.333)>stress(1-(d-u)/hf,551.724138,5333.33333)) b=u;
        else a=u;
    }
    return stress(1-.5*(a+b)/hs,13255.0336,658333.333);
}

int main(int argc,char** argv) { @autoreleasepool {
    if(argc!=3) return 2;
    auto dev=MTLCreateSystemDefaultDevice();NSError* err=nil;
    auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    auto queue=[dev newCommandQueue];
    constexpr unsigned rows=38,cols=rows*4,nv=7,queries=8+rows;
    std::vector<mr_float4> stock(cols);std::vector<unsigned> regions(cols),height(cols);
    std::vector<MRHumanRestingVertexMap> map(cols);std::vector<MRHumanRestingInfluence> influences;
    std::vector<MRNumiHumanStandContactGPU> contacts(rows);std::vector<MRArticulatedPointWorldGPU> points(queries);
    std::vector<float> jac(3*queries*nv);
    for(unsigned p=0;p<8;++p) {
        for(unsigned a=0;a<3;++a) {
            ((float*)&points[p].position)[a]=float(std::sin(0.37*(1+p*3+a)));
            for(unsigned j=0;j<nv;++j)jac[(3*p+a)*nv+j]=float(std::cos(.23*(1+(3*p+a)*nv+j)));
        }
    }
    auto affine=[&](unsigned p,unsigned a,int j,const mr_float4& loc) {
        auto v=[&](unsigned k)->double{return j<0?((float*)&points[p+k].position)[a]:jac[(3*(p+k)+a)*nv+j];};
        return v(0)+loc.x*(v(1)-v(0))+loc.y*(v(2)-v(0))+loc.z*(v(3)-v(0));
    };
    std::vector<std::array<double,3*(nv+1)>> action(rows);
    std::vector<double> refF(rows),refK(rows);
    for(unsigned v=0;v<cols;++v) {
        unsigned r=v/4;regions[v]=r;contacts[r].pointQueryIndex=8+r;contacts[r].planePoint.w=0.001f;
        stock[v]={float(.0004*(1+v%4)),.0024f,float(.009+.0002*(v%6)),0};
        float z=(r==0 || v%4==0)?.001f:float(-.0003*(1+v%7));
        if(r==rows-1 && v%4==3)z=-.008f;
        unsigned bits=std::bit_cast<unsigned>(z);height[v]=bits&(1u<<31)?~bits:bits^(1u<<31);
        double f=r==rows-1?0:stock[v].x*exactPressure(-double(z),stock[v].y,stock[v].z);
        double eps=1e-8;
        if (r!=rows-1 && z<0)refK[r]+=stock[v].x*(exactPressure(-z+eps,stock[v].y,stock[v].z)-exactPressure(-z-eps,stock[v].y,stock[v].z))/(2*eps);
        refF[r]+=f;map[v].firstInfluence=unsigned(influences.size());map[v].influenceCount=2;
        for(unsigned b=0;b<2;++b) {
            MRHumanRestingInfluence term{};term.body.x=b;
            term.positionAndWeight={float(.1*std::sin(1.1*v)),float(.23*std::cos(.3*v+b)),float(.17*b),b?.625f:.375f};
            influences.push_back(term);
            for(unsigned a=0;a<3;++a) for(unsigned j=0;j<=nv;++j)
                action[r][a*(nv+1)+j]+=f*term.positionAndWeight.w*affine(4*b,a,j==nv?-1:int(j),term.positionAndWeight);
        }
    }
    auto keys=buf(dev,height),cb=buf(dev,contacts),pb=buf(dev,points),lb=buf(dev,std::vector<mr_float4>(queries)),jb=buf(dev,jac);
    MRHumanRestingSupportDispatchGPU d{};d.counts={cols,rows,nv,1};d.strides={2,0,queries,queries*3*nv};
    {
        NumiHumanResolvedBed owner(dev,lib,stock,map,influences,regions,rows,1,argv[2]);
        auto command=[queue commandBuffer];owner.encode(command,keys,cb,pb,lb,jb,d,17);
        [command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)return 3;
    }
    auto out=static_cast<MRNumiHumanStandContactGPU*>(cb.contents);auto posed=static_cast<MRArticulatedPointWorldGPU*>(pb.contents);
    auto derivative=static_cast<float*>(jb.contents);
    double forceError=0,workError=0,tangentError=0;
    for(unsigned r=1;r<rows-1;++r) {
        if(!std::isfinite(out[r].planePoint.x)||!std::isfinite(out[r].planePoint.y)) return 5;
        forceError=std::max(forceError,std::abs(out[r].planePoint.x/refF[r]-1));
        tangentError=std::max(tangentError,std::abs(out[r].planePoint.y/refK[r]-1));
        unsigned q=8+r;
        for(unsigned a=0;a<3;++a)for(unsigned j=0;j<=nv;++j) {
            float result=j==nv?((float*)&posed[q].position)[a]:derivative[(3*q+a)*nv+j];
            if(!std::isfinite(result))return 6;
            workError=std::max(workError,std::abs(result*out[r].planePoint.x-action[r][a*(nv+1)+j])/refF[r]);
        }
    }
    std::cout<<std::setprecision(12)<<"{\"max_relative_force_error\":"<<forceError<<",\"max_relative_tangent_error\":"<<tangentError
        <<",\"max_force_normalized_transfer_error\":"<<workError<<",\"open_row_N\":"<<out[0].planePoint.x
        <<",\"excess_strain_propagated\":"<<(std::isnan(out[rows-1].planePoint.x)?"true":"false")<<"}\n";
    return (forceError<.001 && tangentError<.004 && workError<.001 && out[0].planePoint.x==0 && std::isnan(out[rows-1].planePoint.x))?0:4;
}}
