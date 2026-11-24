#include <metal_stdlib>
using namespace metal;
struct Span { uint offset, count, mask, id; };
struct Tile { float mechanics2, pressure2, maximum2; uint row; };
kernel void summarize_certificate_rows(device const float4 *residual [[buffer(0)]],
                                      constant Span *spans [[buffer(1)]],
                                      device Tile *out [[buffer(2)]],
                                      uint group [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint tiles=128;
    const Span s=spans[group/tiles];
    float m=0,p=0,highest=-1;
    uint witness=UINT_MAX;
    for(uint i=(group%tiles)*32+lane;i<s.count;i+=32*tiles) {
        float4 r=residual[s.offset+i];
        float v=(s.mask==1)?r.x*r.x:dot(r.xyz,r.xyz);
        float w=(s.mask==15)?r.w*r.w:0;
        m+=v;p+=w;
        if(v+w>highest){highest=v+w;witness=i;}
    }
    const float peak=simd_max(highest);
    const uint row=simd_min(peak==highest?witness:UINT_MAX);
    const float M=simd_sum(m),P=simd_sum(p);
    if(!lane)out[group]={M,P,peak,row};
}
