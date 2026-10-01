// Independent probe: links the pinned, unmodified FEBio 3.0 material equations.
// No Matter headers, formula implementations, or native material expressions.
#include <FEBioLib/febio.h>
#include <FECore/FEModel.h>
#include <FECore/FECoreKernel.h>
#include <FECore/FEModelParam.h>
#include <FEBioMech/FETransIsoMooneyRivlin.h>
#include <FEBioMech/FEMooneyRivlin.h>
#include <FEBioMech/FEPreStrainUncoupledElastic.h>
#include <FEBioMech/FEInSituStretchGradient.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdexcept>
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static void scalar(FECoreBase& m,const char* key,double v){
    FEParam* p=m.FindParameter(ParamString(key));require(p,"missing source scalar");p->value<double>()=v;
}
static mat3d matrix(const double* v){return mat3d(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]);}
int main(int argc,char** argv){try{
    require(argc==3,"usage: febio-material-oracle INPUT OUTPUT");
    febio::InitLibrary();FEModel model;
    std::ifstream input(argv[1]);std::ofstream output(argv[2]);
    require(input.good()&&output.good(),"cannot open material rows");output<<std::setprecision(17);
    std::string line;unsigned count=0;double maxFD=0;
    while(std::getline(input,line)){
        std::istringstream row(line);double v[29];
        for(double& x:v)require(bool(row>>x)&&std::isfinite(x),"invalid material row");
        std::string extra;require(!(row>>extra),"extra material field");
        require(v[7]==0||v[7]==1,"unsupported fibre scale");
        vec3d a(v[8],v[9],v[10]);require(std::abs(a.norm()-1)<1e-10,"nonunit fibre");
        vec3d seed=std::abs(a.x)<=std::abs(a.y)&&std::abs(a.x)<=std::abs(a.z)?vec3d(1,0,0):
            (std::abs(a.y)<=std::abs(a.z)?vec3d(0,1,0):vec3d(0,0,1));
        vec3d b=seed-a*(a*seed);b.unit();vec3d c=a^b;
        mat3d Q(a.x,b.x,c.x,a.y,b.y,c.y,a.z,b.z,c.z);
        std::unique_ptr<FEUncoupledMaterial> material;
        if(v[7]==1){
            auto* wrapper=fecore_alloc(FEPreStrainUncoupledElastic,&model);
            auto* elastic=fecore_alloc(FETransIsoMooneyRivlin,&model);
            auto* prestrain=fecore_alloc(FEInSituStretchGradient,&model);
            require(wrapper&&elastic&&prestrain,"source factory missing");
            material.reset(wrapper);
            require(wrapper->SetProperty(wrapper->FindPropertyIndex("elastic"),elastic),"elastic property failed");
            require(wrapper->SetProperty(wrapper->FindPropertyIndex("prestrain"),prestrain),"prestrain property failed");
            elastic->c1=v[0];elastic->c2=0;
            scalar(*elastic,"c3",v[2]);scalar(*elastic,"c4",v[3]);scalar(*elastic,"c5",v[4]);scalar(*elastic,"lam_max",v[5]);
            prestrain->m_lam=v[6];prestrain->m_biso=true;
        }else{
            require(v[6]==1,"cartilage prestrain unsupported");
            auto* elastic=fecore_alloc(FEMooneyRivlin,&model);require(elastic,"cartilage factory missing");
            elastic->m_c1=v[0];elastic->m_c2=0;material.reset(elastic);
        }
        material->m_K=v[1];
        auto* frame=material->FindParameter(ParamString("mat_axis"));require(frame,"source frame missing");
        frame->value<FEParamMat3d>()=Q;
        std::unique_ptr<FEMaterialPoint> point(material->CreateMaterialPointData());
        require(bool(point),"source point creation failed");point->Init();
        auto* ep=point->ExtractData<FEElasticMaterialPoint>();require(ep,"elastic point missing");
        const mat3d F=matrix(v+11),dF=matrix(v+20);require(F.det()>0,"inverted deformation");
        ep->m_F=F;ep->m_J=F.det();ep->m_s=material->Stress(*point);
        const mat3ds sigma=ep->m_s;const tens4ds tangent=material->Tangent(*point);
        const mat3d inverse=F.inverse(),l=dF*inverse;
        const mat3d P=(sigma*inverse.transpose())*F.det();
        const mat3d dP=((mat3d(tangent.dot(l.sym()))+l*sigma)*inverse.transpose())*F.det();
        mat3d samples[2];
        for(int side=0;side<2;++side){
            ep->m_F=F+dF*(side==0?1e-6:-1e-6);ep->m_J=ep->m_F.det();
            samples[side]=(material->Stress(*point)*ep->m_F.inverse().transpose())*ep->m_J;
        }
        double err=0,denom=0;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            double fd=(samples[0](i,j)-samples[1](i,j))/2e-6;
            err+=std::pow(dP(i,j)-fd,2);denom+=fd*fd;
            require(std::isfinite(P(i,j))&&std::isfinite(dP(i,j)),"nonfinite source response");
            output<<P(i,j)<<' '<<dP(i,j)<<' ';
        }
        maxFD=std::max(maxFD,std::sqrt(err)/std::max(1e6,std::sqrt(denom)));
        output<<'\n';++count;
    }
    output.flush();require(count>0&&output.good(),"empty or failed export");
    std::cout<<"source_material_rows="<<count<<" maximum_source_tangent_fd_error="<<maxFD
             <<" energy_output=unqualified assembled_equilibrium=unqualified\n";
    require(maxFD<1e-5,"source tangent failed finite difference");return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
