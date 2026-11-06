// Human material precision diagnostic. Executes existing production Matter kernels.
// Compile with NUMI_HUMAN_FEM_REFERENCE_OWNER pointing to the existing owner helper.
#ifndef NUMI_HUMAN_FEM_REFERENCE_OWNER
#error "Define NUMI_HUMAN_FEM_REFERENCE_OWNER"
#endif
#define main numi_existing_fem_reference_main
#include NUMI_HUMAN_FEM_REFERENCE_OWNER
#undef main
#include "numi/matter/canonical_neo_hookean.h"

int main(int argc,char** argv){@autoreleasepool{try{
    need(argc==3,"usage: human-skin-fat-precision SKIN.nmatter FAT.nmatter");
    Device device;
    std::cout<<std::setprecision(17)<<"{\"device\":\""<<[device.device name].UTF8String
        <<"\",\"role\":\"production-kernel precision diagnostic; not anatomical qualification\",\"cases\":[";
    bool first=true;
    double maximumRandomForceError=0, maximumRandomTangentError=0, maximumFiniteDifferenceError=0;
    for(unsigned nativeMode=0;nativeMode<3u;++nativeMode){
        const auto pipeline=[&](NSString* name){
            MTLFunctionConstantValues* values=[MTLFunctionConstantValues new];
            const BOOL enabled=nativeMode==1u ? YES : NO, disabled=NO;
            [values setConstantValue:&enabled type:MTLDataTypeBool atIndex:30u];
            [values setConstantValue:&enabled type:MTLDataTypeBool atIndex:33u];
            [values setConstantValue:&disabled type:MTLDataTypeBool atIndex:34u];
            NSError* error=nil;
            auto function=[device.library newFunctionWithName:name constantValues:values error:&error];
            need(function!=nil,"precision pipeline function missing");
            auto state=[device.device newComputePipelineStateWithFunction:function error:&error];
            need(state!=nil,"precision pipeline creation failed");return state;
        };
        device.forcePipeline=pipeline(@"numi_matter_metal::nm_fem_internal_forces");
        device.operatorPipeline=pipeline(@"numi_matter_metal::nm_fem_apply_operator_elements");
    for(unsigned materialIndex=0;materialIndex<2u;++materialIndex){
        auto parsed=parseMatterFile(argv[materialIndex+1]);
        need(parsed.succeeded(),diagnostic(parsed.diagnostics));
        auto authored=source(parsed.material,parsed.material,true);
        authored.objects[0].femReferenceDisplacementGradient=true;
        authored.objects[0].femMaterialFrameRotations={{{0,0,0,1}},{{0,0,0,1}}};
        authored.objects[0].femNodes=authored.objects[0].femReferenceNodes;
        auto world=cook(authored);
        if(nativeMode==2u)for(auto& m:world.materials){
            m.flags|=nmCanonicalNeoHookeanPrecisionFlag;
            m.flags&=~NM_MATERIAL_CANONICAL_COMPRESSIBLE_NEOHOOKEAN_TANGENT;
        }
        const auto& material=world.materials[0];
        need(material.parameterCount==3u&&material.stateCount==0u,
            "diagnostic requires expected stateless density/shear/bulk material");
        const double shear=world.parameters[material.parameterOffset+1].valueAndBounds.x;
        const double bulk=world.parameters[material.parameterOffset+2].valueAndBounds.x;
        for(double strain:{0.0,1e-9,1e-8,1e-7,1e-6,1e-4,1e-2}){
            auto input=manufactured(world);
            for(unsigned env=0;env<2u;++env)for(unsigned n=0;n<world.dispatch.femNodeCount;++n){
                auto& node=input.nodes[env*world.dispatch.femNodeCount+n];
                const double exact=strain*double(node.restAndFixed.x);
                const float high=static_cast<float>(exact);
                const float low=static_cast<float>(exact-double(high));
                node.referenceDisplacementAndMode={high,0,0,1};
                node.referenceDisplacementLow={low,0,0,0};
                node.positionAndMass.x=mrCompensatedProjectionScalar(node.restAndFixed.x,high,low);
            }
            const auto result=dispatch(device,world,input,false);
            paired(result,"production force environments");
            double error=0,scale=0,observed=0;
            for(unsigned ti=0;ti<world.dispatch.tetrahedronCount;++ti){
                // Binary reference geometry makes eps exact except u's two-word rounding.
                const auto& t=world.fem.tetrahedra[ti];
                const auto ids=std::array<unsigned,4>{t.nodes.x,t.nodes.y,t.nodes.z,t.nodes.w};
                const auto u=[&](unsigned id){const auto& node=input.nodes[id];
                    return Vec{double(node.referenceDisplacementAndMode.x)+node.referenceDisplacementLow.x,0,0};};
                const auto h=ref::multiply(columns(sub(u(ids[1]),u(ids[0])),
                    sub(u(ids[2]),u(ids[0])),sub(u(ids[3]),u(ids[0]))),inverseRest(t));
                const double eps=h[0],a=1.0+eps,logj=std::log1p(eps);
                Matrix stress{};
                stress[0]=shear*(eps+eps/a)+bulk*logj/a;
                stress[4]=stress[8]=bulk*logj;
                const auto expected=forces(t,stress),actual=forces(result[ti]);
                for(unsigned n=0;n<4;++n)for(unsigned k=0;k<3;++k){
                    need(std::isfinite(actual[n][k]),"nonfinite native force");
                    error=std::max(error,std::abs(actual[n][k]-expected[n][k]));
                    scale=std::max(scale,std::abs(expected[n][k]));
                    observed=std::max(observed,std::abs(actual[n][k]));
                }
            }
            if(strain==0)need(observed==0,"nonzero force at exact identity");
            if(strain==1e-2)need(error/std::max(scale,1e-300)<1e-3,"moderate-strain oracle calibration failed");
            if(!first)std::cout<<',';first=false;
            std::cout<<"{\"material\":"<<materialIndex<<",\"native_mode\":"<<nativeMode<<",\"flags\":"<<material.flags
                <<",\"shear_Pa\":"<<shear<<",\"bulk_Pa\":"<<bulk
                <<",\"strain\":"<<strain<<",\"expected_max_N\":"<<scale
                <<",\"observed_max_N\":"<<observed<<",\"max_error_N\":"<<error
                <<",\"relative_max_error\":"<<(scale?error/scale:0)<<'}';
        }
        // Deterministic random affine fields and directions; these are independent
        // of the nine basis columns used by the production tangent cache.
        std::uint32_t rng=0x7265616cu;
        const auto sample=[&](){rng=1664525u*rng+1013904223u;return double(rng)/4294967296.0-.5;};
        for(unsigned trial=0;trial<12u;++trial){
            auto input=manufactured(world);
            Matrix h{},direction{};
            const double amplitude=trial<4u?1e-8:trial<8u?1e-4:.1;
            for(unsigned i=0;i<9u;++i){h[i]=amplitude*sample();direction[i]=sample();}
            for(unsigned env=0;env<2u;++env)for(unsigned n=0;n<world.dispatch.femNodeCount;++n){
                auto& node=input.nodes[env*world.dispatch.femNodeCount+n];
                const auto rest=xyz(node.restAndFixed),u=transform(h,rest),dx=transform(direction,rest);
                float* high[3]={&node.referenceDisplacementAndMode.x,&node.referenceDisplacementAndMode.y,&node.referenceDisplacementAndMode.z};
                float* low[3]={&node.referenceDisplacementLow.x,&node.referenceDisplacementLow.y,&node.referenceDisplacementLow.z};
                float* position[3]={&node.positionAndMass.x,&node.positionAndMass.y,&node.positionAndMass.z};
                for(unsigned axis=0;axis<3;++axis){
                    *high[axis]=float(u[axis]);*low[axis]=float(u[axis]-double(*high[axis]));
                    *position[axis]=mrCompensatedProjectionScalar(float(rest[axis]),*high[axis],*low[axis]);
                }
                node.referenceDisplacementAndMode.w=1;
                input.direction[env*world.dispatch.femNodeCount+n]={float(dx[0]),float(dx[1]),float(dx[2]),0};
            }
            const Oracle oracle=[&](unsigned,const Matrix& f,const Matrix& df,const Matrix&,const Matrix&){
                const Matrix a=ref::sub(f,ref::identity),fit=ref::inverseTranspose(f);
                const double jm=a[0]+a[4]+a[8]+a[0]*a[4]+a[0]*a[8]+a[4]*a[8]
                    -a[1]*a[3]-a[2]*a[6]-a[5]*a[7]+ref::determinant(a);
                const double logj=std::log1p(jm),v=ref::contract(fit,df);
                ref::Result out;
                out.stress=ref::add(ref::scale(ref::add(a,ref::multiply(fit,ref::transpose(a))),shear),ref::scale(fit,bulk*logj));
                out.tangent=ref::add(ref::add(ref::scale(df,shear),ref::scale(fit,bulk*v)),
                    ref::scale(ref::multiply(ref::multiply(fit,ref::transpose(df)),fit),shear-bulk*logj));
                return out;
            };
            const auto force=dispatch(device,world,input,false),tangent=dispatch(device,world,input,true);
            paired(force,"random force");paired(tangent,"random tangent");
            const double fe=compare(world,input,force,false,oracle),te=compare(world,input,tangent,true,oracle),
                fd=finiteDifference(device,world,input,tangent);
            maximumRandomForceError=std::max(maximumRandomForceError,fe);
            maximumRandomTangentError=std::max(maximumRandomTangentError,te);
            maximumFiniteDifferenceError=std::max(maximumFiniteDifferenceError,fd);
            need(fe<1e-5&&te<1e-5&&fd<1e-4,"random force/tangent or finite-difference gate failed: "+number(fe)+","+number(te)+","+number(fd));
        }
    }}
    std::cout<<"],\"identity_and_moderate_strain_calibration\":true"
        <<",\"random_force_max_relative_error\":"<<maximumRandomForceError
        <<",\"random_tangent_max_relative_error\":"<<maximumRandomTangentError
        <<",\"finite_difference_max_relative_error\":"<<maximumFiniteDifferenceError<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<"human skin/fat precision diagnostic failed: "<<e.what()<<'\n';return 1;}}}
