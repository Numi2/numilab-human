#include <simd/simd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using simd::float2;
using simd::float3;
using simd::float4;
using simd::cross;
using simd::dot;
static float mix(float a,float b,float t){return a+(b-a)*t;}
static simd_float2 mix(simd_float2 a,simd_float2 b,float t){return a+(b-a)*t;}
static simd_float3 mix(simd_float3 a,simd_float3 b,float t){return a+(b-a)*t;}
static simd_float2 makeFloat2(float a){return {a,a};}
static simd_float2 makeFloat2(float a,float b){return {a,b};}
#define float2(...) makeFloat2(__VA_ARGS__)
#define float4(...) simd_make_float4(__VA_ARGS__)
struct float3x3 {
    float3 col[3];
    float3x3(float3 a,float3 b,float3 c):col{a,b,c}{}
    float3 &operator[](unsigned i){return col[i];}
    const float3 &operator[](unsigned i)const{return col[i];}
};
static float3x3 operator-(float3x3 a,float3x3 b) {return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static float3x3 operator*(float3x3 a,float s) {return {a[0]*s,a[1]*s,a[2]*s};}
static float determinant(float3x3 m){return simd_determinant(simd_matrix(m[0],m[1],m[2]));}
static simd_double3 wide(float3 v){return __builtin_convertvector(v,simd_double3);}
using std::copysign;
using std::fma;
using std::isfinite;
using std::max;
using std::sqrt;
using uint = unsigned;

static float3x3 cofactorMatrix(float3x3 m) {
    return float3x3(cross(m[1],m[2]),cross(m[2],m[0]),cross(m[0],m[1]));
}
#include "swept_material_direction.metalinc"

static void check(bool ok) {
    if (!ok) throw std::runtime_error("Swept-volume regression");
}
static double det3(simd_double3 a,simd_double3 b,simd_double3 c) {
    return dot(a,cross(b,c));
}
static double controlValue(float3x3 start, float3x3 end, unsigned term, float3x3 dir, double delta) {
    simd_double3 s[3], e[3];
    for (unsigned i=0;i<3;++i) {
        s[i]=wide(start[i]);
        e[i]=wide(end[i])+delta*wide(dir[i]);
    }
    if (term==1) return (det3(e[0],s[1],s[2])+det3(s[0],e[1],s[2])+det3(s[0],s[1],e[2]))/3.;
    if (term==2) return (det3(s[0],e[1],e[2])+det3(e[0],s[1],e[2])+det3(e[0],e[1],s[2]))/3.;
    return det3(e[0],e[1],e[2]);
}
int main() {
    const float3x3 id(simd_make_float3(1,0,0),simd_make_float3(0,1,0),simd_make_float3(0,0,1));
    auto flip=id;flip[0]*=-1.f;flip[1]*=-1.f;
    check(registeredSweptControl(id,flip,.35f,2.5f).value<.35f);
    auto a=id,b=id;a[0]*=2.f;a[1]*=.5f;b[1]*=2.f;b[0]*=.5f;
    check(registeredSweptControl(a,b,.35f,1.4f).value>1.4f);
    check(registeredSweptControl(id,id,.35f,2.5f).term==3u);
    std::mt19937 rng(78);
    std::uniform_real_distribution<float> draw(-1.f,1.f);
    double worst=0;unsigned tested=0,interior=0;
    for(unsigned i=0;i<16000;++i) {
        auto start=id,end=id,direction=id;
        for(unsigned j=0;j<3;++j)for(unsigned k=0;k<3;++k) {
            start[j][k]+=.65f*draw(rng);end[j][k]+=1.3f*draw(rng);direction[j][k]=draw(rng);
        }
        for(unsigned term=1;term<=3;++term) {
            const float3x3 g=registeredControlGradient(start,end,term);
            const double analytic=registeredTensorProduct(g,direction);
            const double numeric=(controlValue(start,end,term,direction,1e-5)-controlValue(start,end,term,direction,-1e-5))/2e-5;
            worst=max(worst,std::abs(analytic-numeric)/max(1.,std::abs(numeric)));
        }
        if (determinant(start)<.35f||determinant(start)>2.5f||determinant(end)<.35f||determinant(end)>2.5f)continue;
        ++tested;
        auto selected=registeredSweptControl(start,end,.35f,2.5f);
        double violation=-1.;
        for(unsigned k=0;k<=96;++k) {
            const double t=k/96.;simd_double3 cols[3];
            for(unsigned c=0;c<3;++c)cols[c]=wide(start[c])*(1.-t)+t*wide(end[c]);
            double j=det3(cols[0],cols[1],cols[2]);
            violation=max(violation,max(.35-j,j-2.5));
        }
        if(violation>1e-5){++interior;check(selected.term!=3u);}
        if(violation<-.002)check(selected.term==3u);
    }
    check(worst<3e-6&&interior>10);
    std::cout<<"valid_endpoint_pairs="<<tested<<" interior_violations="<<interior<<" worst_gradient_error="<<worst<<'\n';
}
