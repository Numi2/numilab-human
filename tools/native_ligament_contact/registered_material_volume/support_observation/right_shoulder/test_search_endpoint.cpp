#include <simd/simd.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
using simd::float3;
using simd::dot;
using std::min;
using std::max;
using std::abs;
struct float3x3 { simd_float3x3 m; const float3& operator[](size_t i) const { return m.columns[i]; } };
static float determinant(float3x3 m) { return simd_determinant(m.m); }
static float3 abs(float3 v) { return simd_make_float3(std::abs(v.x),std::abs(v.y),std::abs(v.z)); }
struct NMRegisteredVolumeControl { float value; unsigned term; float location; };
#include "search_endpoint.metalinc"

int main() {
    const float3x3 I{matrix_identity_float3x3};
    size_t held=0;
    for(float bound : {.35f,2.5f}) {
        const float toward=bound==.35f?INFINITY:-INFINITY;
        for(unsigned steps=1;steps<=128;++steps) {
            float q=bound;
            for(unsigned k=0;k<steps;++k) q=std::nextafter(q,toward);
            auto b=I; b.m.columns[2].z=q;
            auto result=registeredEndpointControl(I,b,.35f,2.5f,{q,3u,1.f});
            assert(result.value==bound || result.value==q);
            if(result.value==bound) ++held;
            auto a=b; a.m.columns[2].z=std::nextafter(q,bound);
            assert(registeredEndpointControl(a,b,.35f,2.5f,{q,3u,1.f}).value==q);
            assert(registeredEndpointControl(b,b,.35f,2.5f,{q,3u,1.f}).value==q);
        }
    }
    auto b=I;b.m.columns[2].z=.9f;
    assert(registeredEndpointControl(I,b,.35f,2.5f,{.9f,3u,1.f}).value==.9f);
    assert(held>0 && held<256);
    std::printf("outward_near_boundary_cases=%zu inward_and_stationary_cases=512\n",held);
}
