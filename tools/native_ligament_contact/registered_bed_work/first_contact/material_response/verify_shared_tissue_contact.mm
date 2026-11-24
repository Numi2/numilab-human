#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/fem_body_triangle_contact_sample.hpp"
#include "metalrobo/numi_human_tendon_gpu.h"
#include <cmath>
#include <iostream>
#include <vector>
using namespace numi::matter::fem_body_triangle_contact;
using Row=NMNumiHumanFEMBodyContactSampleGPU;
using Audit=NMNumiHumanArticularContactAuditGPU;
static void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
int main(int argc,char**argv){@autoreleasepool{
    check(argc==2,"metallib path required");
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();check(dev!=nil,"GPU unavailable");
    NSError* error=nil;
    auto lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    auto fn=[lib newFunctionWithName:@"numi_matter_metal::nm_numi_human_audit_fem_body_contact"];
    check(fn!=nil,"registered production audit missing");
    auto pipeline=[dev newComputePipelineStateWithFunction:fn error:&error];
    check(pipeline!=nil,"native certificate kernel unavailable");
    auto queue=[dev newCommandQueue];
    auto observe=[&](std::vector<Row> rows,float depth){
        NMNumiHumanTendonFEMLoadDispatchGPU d{};
        d.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;d.environmentCount=1;
        d.bodyPoseStride=1;d.femBodyContactSampleCount=rows.size();d.femNodeCount=1;
        NMFEMNodeStateGPU point{};point.positionAndMass={0.002f,0.002f,-depth,1.0f};
        MRArticulatedBodyPoseGPU pose{};pose.orientation.w=1;
        id<MTLBuffer> buffers[9]{};
        auto copy=[&](const void* bytes,size_t n){return [dev newBufferWithBytes:bytes length:n options:MTLResourceStorageModeShared];};
        buffers[0]=copy(&d,sizeof(d));buffers[1]=copy(rows.data(),rows.size()*sizeof(Row));
        buffers[2]=copy(&point,sizeof(point));buffers[3]=copy(&pose,sizeof(pose));
        for(unsigned k=4;k<9;++k){buffers[k]=[dev newBufferWithLength:1024 options:MTLResourceStorageModeShared];memset(buffers[k].contents,0,1024);}
        auto command=[queue commandBuffer];auto enc=[command computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        for(unsigned k=0;k<9;++k)[enc setBuffer:buffers[k] offset:0 atIndex:k];
        [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];[command commit];[command waitUntilCompleted];
        check(command.status==MTLCommandBufferStatusCompleted,"certificate command failed");
        Audit a;memcpy(&a,buffers[4].contents,sizeof(a));return a;
    };
    auto make=[](float area,float k,float inverseE){return makeTriangleContactSample(
        0,0,{0,0,0,0},{.01f,0,0,0},{0,.01f,0,0},area,k,inverseE);};
    const Row skin=make(.000002f,39500.f/.002f,1.f/39500.f);
    const Row fat=make(.000007f,1600.f/.001f,1.f/1600.f);
    auto area=[](Row s){return s.bodyLocalPointAndArea.w;};
    auto stiffness=[](Row s){return s.bodyLocalNormalAndReferenceSeparation.w;};
    auto compliance=[](Row s){return s.stiffnessAndNormalStrainPerPressure.w;};
    double worst=0;unsigned compared=0,positive=0;
    for(auto first:{skin,fat})for(auto second:{fat,skin}){
        double a=double(area(first))+area(second);
        float k=float((double(area(first))*stiffness(first)+double(area(second))*stiffness(second))/a);
        const float peak=std::max(stiffness(first),stiffness(second));
        Row sum=make(float(a),k,float(std::max(double(stiffness(first))*compliance(first),double(stiffness(second))*compliance(second))/k));
        sum.reserved0=k==peak?0u:std::bit_cast<unsigned>(-peak);check(validTriangleContactSample(sum),"invalid mixture");
        for(float depth:{-.001f,0.f,1.e-6f,.0002f,.002f}){
            Audit original=observe({first,second},depth),merged=observe({sum},depth);
            check(original.energyStrainClosureAndAccepted.w==merged.energyStrainClosureAndAccepted.w,"different certificate");
            const auto slots=[](const Audit& a){return std::array<float,8>{a.normalForceAreaAndCounts.x,a.normalForceAreaAndCounts.y,
                a.momentResidualAndMaximumPressure.w,a.energyStrainClosureAndAccepted.x,a.energyStrainClosureAndAccepted.y,
                a.energyStrainClosureAndAccepted.z,a.forceResidualAndL1.x,a.momentResidualAndMaximumPressure.y};};
            auto reference=slots(original),current=slots(merged);
            for(unsigned i=0;i<current.size();++i){double rel=std::abs(double(current[i])-reference[i])/std::max(1e-15,std::abs(double(reference[i])));check(std::isfinite(rel)&&rel<8e-6,"constituent mismatch");worst=std::max(worst,rel);++compared;}
            if(depth>0){check(current[0]>0&&current[1]>0&&current[2]>0&&current[3]>0&&current[4]>0,"missing physical response");++positive;}
        }
    }
    for(unsigned word:{1u,0x4e484331u,0xff800000u,0x7fc00000u,std::bit_cast<unsigned>(-10.f)}){
        auto invalid=skin;invalid.reserved0=word;check(!validTriangleContactSample(invalid),"host admitted invalid mixture");
        if(word!=0x4e484331u){auto a=observe({invalid},.0002f);check(!std::isfinite(a.forceResidualAndL1.x),"GPU admitted invalid mixture");}
    }
    std::cout<<"GPU="<<dev.name.UTF8String<<" comparisons="<<compared<<" loaded_configurations="<<positive<<" maximum_relative_error="<<worst<<" rejected_invalid_payloads=5\n";
}}
