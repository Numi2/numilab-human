#include "numi/matter/fem_compensated_deformation.h"
#include <array>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
using V=std::array<double,3>;
double det(const double* a){return a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);}
int main(int argc,char**argv){
 if(argc!=2)return 64;
 std::ifstream f(argv[1],std::ios::binary);uint64_t n=0;f.read((char*)&n,8);
 if(!f||!n||n>20000)return 2;
 std::cout<<std::setprecision(17)<<"{\"schema\":\"h706.shared-reference-deformation-diagnostic.v1\",\"physical_admission\":false,\"native_steps\":0,\"gpu_executed\":false,\"selected_tets\":"<<n<<",\"cases\":[";
 for(int test=0;test<3;++test){
  f.clear();f.seekg(8);double minJ=1e300,maxJ=-1e300,maxError=0;size_t invalid=0;
  double theta=(test==0?0.0:(test==1?-1.0:1.0))*std::acos(-1.0)/180.0,c=std::cos(theta),s=std::sin(theta);
  const double R[9]={c,-s,0,s,c,0,0,0,1};
  for(uint64_t k=0;k<n;++k){
   float X[12];f.read((char*)X,sizeof(X));if(!f)return 3;
   double A[9];for(int r=0;r<3;++r)for(int col=0;col<3;++col)A[3*r+col]=double(X[3*(col+1)+r])-X[r];
   double d=det(A);if(!(d>0))return 4;
   const double invD[9]={
    (A[4]*A[8]-A[5]*A[7])/d,(A[2]*A[7]-A[1]*A[8])/d,(A[1]*A[5]-A[2]*A[4])/d,
    (A[5]*A[6]-A[3]*A[8])/d,(A[0]*A[8]-A[2]*A[6])/d,(A[2]*A[3]-A[0]*A[5])/d,
    (A[3]*A[7]-A[4]*A[6])/d,(A[1]*A[6]-A[0]*A[7])/d,(A[0]*A[4]-A[1]*A[3])/d};
   float inv[9],hi[12],lo[12],Fhi[9],Flo[9],identity[9]={1,0,0,0,1,0,0,0,1};
   for(int j=0;j<9;++j)inv[j]=float(invD[j]);
   for(int node=0;node<4;++node)for(int r=0;r<3;++r){
    double u= test==0 && r==2 ? .001 : 0.;
    if(test!=0)for(int col=0;col<3;++col)u+=(R[3*r+col]-(r==col?1.:0.))*double(X[3*node+col]);
    hi[3*node+r]=float(u);lo[3*node+r]=float(u-double(hi[3*node+r]));
   }
   nmFEMReferenceDeformationWords(hi,lo,inv,identity,Fhi,Flo);
   double F[9];for(int j=0;j<9;++j){F[j]=Fhi[j];maxError=std::max(maxError,std::abs(F[j]-R[j]));}
   double J=det(F);minJ=std::min(minJ,J);maxJ=std::max(maxJ,J);if(!std::isfinite(J)||J<=0)++invalid;
  }
  if(test)std::cout<<",";
  std::cout<<"{\"case\":"<<test<<",\"minimum_J\":"<<minJ<<",\"maximum_J\":"<<maxJ<<",\"nonpositive_or_nonfinite_J\":"<<invalid<<",\"maximum_entry_error_from_R\":"<<maxError<<"}";
 }
 std::cout<<"]}\n";
}
