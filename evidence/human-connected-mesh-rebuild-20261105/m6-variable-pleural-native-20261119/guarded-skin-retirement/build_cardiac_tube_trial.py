"""Assemble a fail-closed native trial using only proved cardiac pressure intervals."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from fit_cardiac_volume_grid import hash_file
from verify_cardiac_tube_membership import read_tubes, ATLAS

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"resting-clearance-and-atrium-002"
SHADER=HERE.parent/"coupled-thumb-reference-001"
WORK=HERE/"cardiac-split-tube-native-build"
EXPECTED_GEOMETRY="3b0acb6146358f3302f2b85e2214bafa9f2fae7b0e51af70fd0838d6609da4a4"
EXPECTED_MAP="0d88db155f7dfe9a93030badf6e570f49050164a7d0aa0ae011065aeada7d3da"


def replace(text, old, new):
    actual=text.count(old)
    if actual!=1:
        raise ValueError(f"expected one match, found {actual}: {old[:100]}")
    return text.replace(old,new)


def bound_table():
    tables=list((HERE/"cardiac-conditioned-live-audit").glob("*.log"))
    tables.extend(HERE.glob("refined-live-window-*.log"))
    passing=set()
    for p in tables:
        for line in p.read_text().splitlines():
            v=list(map(int,line.split()))
            assert len(v)==9
            if not any(v[2:7]):
                passing.add(v[0])
    assert 0 in passing
    boxes=read_tubes()
    indices=sorted(passing)
    def cell(x):
        return float(x).hex()+"f"
    def vec(r):
        return "float4("+", ".join(map(cell,r))+")"
    def bounds(b):
        return "{"+", ".join(vec(r) for r in b)+"}"
    rows=[]
    for i in indices:
        tube=boxes[i]
        # Preserve IDs in unused padding so the resulting receipt can trace
        # a successful lane to the precise certified authoring interval.
        tube[1,7]=i
        rows.append("    {"+", ".join(bounds(tube[n:n+1].reshape(2,4)) for n in range(4))+"}")
    shader="constant CardiacCertifiedTube cardiacCertifiedTubes[] = {\n"+",\n".join(rows)+"\n};\n"
    (WORK/"cardiac_proved_table.metalinc").write_text(shader)
    return indices


def patch_metal():
    code=(SHADER/"human_respiration.metal").read_text()
    for f in ("cardiac_certified_tube.metalinc",):
        (WORK/f).write_bytes((HERE/f).read_bytes())
    code=replace(code,'#include "NumiHumanRestingCommonField.metalinc"',
        '#include "NumiHumanRestingCommonField.metalinc"\n#include "cardiac_certified_tube.metalinc"\n'
        '#include "cardiac_proved_table.metalinc"')
    solve='''    const uint status=nmHumanRestingCommonSolveSimd(
        parameters,boxes,targets,coordinates,lane,iterations,matched,residual);
    if(lane!=0u) return;'''
    code=replace(code,solve,'''    uint status=nmHumanRestingCommonSolveSimd(
        parameters,boxes,targets,coordinates,lane,iterations,matched,residual);
    if(lane!=0u) return;
    if(status==5u) {
        for(uint i=0u;i<sizeof(cardiacCertifiedTubes)/sizeof(cardiacCertifiedTubes[0]);++i) {
            if(cardiacInsideCertifiedTube(cardiacCertifiedTubes[i],coordinates)) {
                matched=parameters.countsAndFlags.x+uint(cardiacCertifiedTubes[i].last[1].w);
                status=0u;
                break;
            }
        }
    }''')
    source=(HERE/"cardiac_compensated_field.metal").read_text()
    funcs=source[source.index("static inline float3 errorFreeAdd"):source.index("kernel void pose_common_field")]
    code=replace(code,"kernel void nm_human_resting_skin(",funcs+"kernel void nm_human_resting_skin(")
    header='''    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(12)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.x)return;
    auto m=map[i];if(!m.influenceCount)return;'''
    code=replace(code,header,'''    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(12)]],
    device float4* splitLow [[buffer(13)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.x)return;
    auto m=map[i];if(!m.influenceCount)return;''')
    code=replace(code,"                for(uint c=0;c<7;++c) local+=basis.displacement[c].xyz*coordinate[c];",
       '''                if(m.deformationWeight.w==2.0f) {
                    if(!exactBodyLocal) local=float3(NAN);
                    else {
                        float3 low=0;
                        for(uint c=0;c<7;++c) addBasis(local,low,basis.displacement[c].xyz,coordinate[c]);
                        splitLow[i]=float4(low,1);
                    }
                }else {
                    for(uint c=0;c<7;++c) local+=basis.displacement[c].xyz*coordinate[c];
                }''')
    code=replace(code,"kernel void nm_human_resting_cardiac_wall_normals(",'''inline float3 splitDifference(float3 a,float3 aLow,float3 b,float3 bLow) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    float3 residual;
    const float3 high=errorFreeAdd(a,-b,residual);
    return high+(residual+(aLow-bLow));
}

kernel void nm_human_resting_cardiac_wall_normals(''')
    code=replace(code,"    device MRVisualVertexGPUV2* vertices [[buffer(4)]],uint i [[thread_position_in_grid]]) {",
        "    device MRVisualVertexGPUV2* vertices [[buffer(4)]],\n"
        "    device const float4* splitLow [[buffer(5)]],uint i [[thread_position_in_grid]]) {")
    code=replace(code,"        normal+=cross(b-a,c-a);","""        const float4 al=splitLow[indices[at]],bl=splitLow[indices[at+1]],
            cl=splitLow[indices[at+2]];
        const bool split=al.w!=0&&bl.w!=0&&cl.w!=0;
        normal+=split?cross(splitDifference(b,bl.xyz,a,al.xyz),
                            splitDifference(c,cl.xyz,a,al.xyz)):cross(b-a,c-a);""")
    prep='''struct CardiacTriangleAudit {
    float3 area;
    bool split;
};
inline CardiacTriangleAudit cardiacTriangleArea(
    float3 pa,float3 pb,float3 pc,float4 al,float4 bl,float4 cl) {
    const bool split=al.w!=0&&bl.w!=0&&cl.w!=0;
    if(split) return {cross(splitDifference(pb,bl.xyz,pa,al.xyz),
                           splitDifference(pc,cl.xyz,pa,al.xyz)),true};
    return {cross(pb-pa,pc-pa),false};
}
inline bool cardiacTriangleCollapsed(const CardiacTriangleAudit a,float3 p,float3 q,float3 r) {
    if(a.split) return all(a.area==float3(0));
    return nm_human_exact_binary32_triangle_is_zero(p,q,r);
}
'''
    code=replace(code,"kernel void nm_human_resting_audit_volume_partials(",prep+"\nkernel void nm_human_resting_audit_volume_partials(")
    code=replace(code,"    device MRHumanRestingVolumeAuditPartialGPU* partials [[buffer(5)]],",
         "    device MRHumanRestingVolumeAuditPartialGPU* partials [[buffer(5)]],\n"
         "    device const float4* splitLow [[buffer(6)]],")
    code=replace(code,"    const float3 origin=vertices[indices[owner.x]].position.xyz;",
         "    const float3 origin=vertices[indices[owner.x]].position.xyz;\n"
         "    const float4 originLow=splitLow[indices[owner.x]];")
    code=replace(code,'''        const float3 u=pb-pa,v=pc-pa;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const bool collapsed=!bad&&nm_human_exact_binary32_triangle_is_zero(pa,pb,pc);
        if(bad||collapsed) {++invalidCount;firstInvalid=min(firstInvalid,triangle);}
        const float3 a=pa-origin,b=pb-origin,c=pc-origin;''','''        const float4 al=splitLow[indices[j]],bl=splitLow[indices[j+1u]],cl=splitLow[indices[j+2u]];
        const auto audit=cardiacTriangleArea(pa,pb,pc,al,bl,cl);
        const bool bad=!all(isfinite(audit.area));
        const bool collapsed=!bad&&cardiacTriangleCollapsed(audit,pa,pb,pc);
        if(bad||collapsed) {++invalidCount;firstInvalid=min(firstInvalid,triangle);}
        const float3 a=audit.split?splitDifference(pa,al.xyz,origin,originLow.xyz):pa-origin;
        const float3 b=audit.split?splitDifference(pb,bl.xyz,origin,originLow.xyz):pb-origin;
        const float3 c=audit.split?splitDifference(pc,cl.xyz,origin,originLow.xyz):pc-origin;''')
    code=replace(code,'''    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(12)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y)return;''','''    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(12)]],
    device const float4* splitLow [[buffer(13)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y)return;''')
    code=replace(code,'''        const float3 u=pb-pa,v=pc-pa;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const uint areaFailure=bad?MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:
            (nm_human_exact_binary32_triangle_is_zero(pa,pb,pc)?''','''        const auto audit=cardiacTriangleArea(pa,pb,pc,splitLow[ids.x],splitLow[ids.y],splitLow[ids.z]);
        const bool bad=!all(isfinite(audit.area));
        const uint areaFailure=bad?MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:
            (cardiacTriangleCollapsed(audit,pa,pb,pc)?''')
    code=replace(code,'''    device uint4* partials [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup uint zero[256]''','''    device uint4* partials [[buffer(3)]],
    device const float4* splitLow [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup uint zero[256]''')
    code=replace(code,'''        const float3 u=b-a,v=c-a;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const bool collapsed=!bad&&nm_human_exact_binary32_triangle_is_zero(a,b,c);''','''        const auto audit=cardiacTriangleArea(a,b,c,splitLow[indices[3u*t]],
            splitLow[indices[3u*t+1u]],splitLow[indices[3u*t+2u]]);
        const bool bad=!all(isfinite(audit.area));
        const bool collapsed=!bad&&cardiacTriangleCollapsed(audit,a,b,c);''')
    code=replace(code,'''    device uint4* result [[buffer(4)]],device MRHumanRestingSurfaceFailureGPU* failure [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {''','''    device uint4* result [[buffer(4)]],device MRHumanRestingSurfaceFailureGPU* failure [[buffer(5)]],
    device const float4* splitLow [[buffer(6)]],uint i [[thread_position_in_grid]]) {''')
    code=replace(code,'''        const float3 u=b-a,v=c-a;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const uint kind=bad?MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:
            (nm_human_exact_binary32_triangle_is_zero(a,b,c)?''','''        const auto audit=cardiacTriangleArea(a,b,c,splitLow[ids.x],splitLow[ids.y],splitLow[ids.z]);
        const bool bad=!all(isfinite(audit.area));
        const uint kind=bad?MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA:
            (cardiacTriangleCollapsed(audit,a,b,c)?''')
    # The production compiler disables unsafe FP transforms. The probe adds
    # per-function pragmas as well for the error-free expansions.
    for marker in ("static inline float3 errorFreeAdd(float3 a, float3 b, thread float3& residual) {",
                   "static inline void addBasis(thread float3& high, thread float3& low, float3 d, float q) {"):
        code=replace(code,marker,marker+"\n#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)")
    (WORK/"human_respiration.metal").write_text(code)


def patch_host(skin_trial):
    code=(SOURCE/"NumiHumanRestingVisual.hpp").read_text()
    # Prior native qualification admitted this exact topology-preserving skin
    # edit with the same physical bed sections and emitted source identities.
    admitted_skin=["07b518d2d49222596060105ea1f7385ab730737af7721a0f6196fd320d6f2862"]
    if skin_trial:
        proof=json.loads((HERE/(skin_trial+"-payload.json")).read_text())
        candidate=HERE/(skin_trial+".nhskin")
        assert hash_file(candidate)==proof["candidate_sha256"]
        assert not proof["new_source_pairs"]
        assert proof["support_anchors_byte_equal"]==32
        admitted_skin.append(proof["candidate_sha256"])
    literals=[f'retainedSourceSkinSHA256 == "{sha}"' for sha in admitted_skin]
    allowed=" ||\n                     ".join(["retainedSourceSkinSHA256 == kSectionCompatibleArmRebind",*literals])
    code=replace(code,"retainedSourceSkinSHA256 == kSectionCompatibleArmRebind &&",
                 "("+allowed+") &&")

    # Reuse the existing verified startup pack path. It remains opt-in and
    # uses the exact initial pack digest and existing topology verification.
    cache=(HERE.parent/"resting-clearance-and-atrium-004/pack-cache-build/NumiHumanRestingVisual.hpp").read_text()
    before="        require(metalrobo::writeVisualAssetPack(pack,packPath,&error),error);"
    first=cache.index("        const char* initialPackCache=")
    last=cache.index("        initialPackPath=packPath;",first)
    next_line="\n        initialPackPath=packPath;initialPackContentHash=pack.contentHash;"
    code=replace(code,before+next_line,cache[first:last].rstrip()+next_line)
    before="        initialPackFileSHA256=loadedKneeSHA256Hex(loadedKneeFileSHA256(packPath));"
    first=cache.index(before)+len(before)
    last=cache.index("        const float distance=",first)
    code=replace(code,before,before+cache[first:last].rstrip())
    code=replace(code,"    id<MTLBuffer> vertexCaptureBuffer=nil;",
        "    id<MTLBuffer> vertexCaptureBuffer=nil,cardiacSplitLow=nil;")
    code=replace(code,
        "        e.setBuffer(e.context,(__bridge void*)self.commonFieldCoordinates,0,12);\n        e.dispatchThreads(e.context,lease.meshVertexCount,64);",
        "        e.setBuffer(e.context,(__bridge void*)self.commonFieldCoordinates,0,12);\n"
        "        e.setBuffer(e.context,(__bridge void*)self.cardiacSplitLow,0,13);\n"
        "        e.dispatchThreads(e.context,lease.meshVertexCount,64);")
    n="            e.setBuffer(e.context,lease.meshIndices,0,3);e.setBuffer(e.context,lease.meshVertices,0,4);"
    assert code.count(n)==2
    code=code.replace(n,n+"\n            e.setBuffer(e.context,(__bridge void*)self.cardiacSplitLow,0,5);")
    for key,num in (("self.volumeAuditPartials,0,5);\n        e.dispatchThreads",6),
                    ("self.commonFieldCoordinates,0,12);\n        e.dispatchThreads(e.context,self.auditCount",13),
                    ("self.meshAuditPartials,0,3);\n        e.dispatchThreads",4),
                    ("self.meshAuditResult,sizeof(mr_uint4),5);\n        e.dispatchThreads",6)):
        before,after=key.split("\n")
        code=replace(code,key,before+f"\n        e.setBuffer(e.context,(__bridge void*)self.cardiacSplitLow,0,{num});\n"+after)
    code=replace(code,"        commonFieldMapBuffer=[device newBufferWithBytes:",
        f'''        require(commonCardiacGeometry&&functional.commonFieldAnatomyPayloadSHA256=="{EXPECTED_GEOMETRY}"&&
                    functional.commonFieldMapSHA256=="{EXPECTED_MAP}",
                "certified cardiac tube binding differs from the proved anatomy or field");
        cardiacSplitLow=[device newBufferWithLength:maps.size()*sizeof(mr_float4) options:MTLResourceStorageModeShared];
        require(cardiacSplitLow!=nil,"could not allocate the split cardiac position buffer");
        std::memset(cardiacSplitLow.contents,0,cardiacSplitLow.length);
        commonFieldMapBuffer=[device newBufferWithBytes:''')
    code=replace(code,"        NSError* jsonError=nil;\n        NSDictionary* receipt=@{",
        '''        cardiacInputs[@"common_position_low"]=retainCardiacInput("common-position-low",
            cardiacSplitLow,pack.vertices.size()*sizeof(mr_float4),"mr_float4");
        NSError* jsonError=nil;
        NSDictionary* receipt=@{''')
    (WORK/"NumiHumanRestingVisual.hpp").write_text(code)


def patch_coupling():
    tubes=read_tubes()
    source=(SOURCE/"NumiHumanRestingCoupling.hpp").read_text()
    marker="// The existing articulated owner remains responsible"
    table=[]
    for tube in tubes:
        row=lambda r: "{"+",".join(float(x).hex()+"f" for x in r[:7])+"}"
        table.append("    {"+",".join(map(row,tube))+", "+str(int(tube[0,7]))+"}")
    check='''// This mirrors the atlas identity for exported GPU states. The GPU alone
// admits a proved tube during simulation. A terminal readback must report
// the GPU's matching domain and satisfy its coordinate bounds.
inline bool numiHumanRestingTerminalDomain(
    std::size_t boxCount,std::uint32_t domain,const std::array<float,7>& q) {
    if(domain<boxCount) return true;
    struct Tube {
        float first[7],last[7],low[7],high[7];
        int pivot;
    };
    static constexpr Tube tubes[]={
TUBE_ROWS
    };
    const auto index=domain-boxCount;
    if(index>=std::size(tubes)) return false;
    const auto& tube=tubes[index];
    const double p0=tube.first[tube.pivot];
    const double phase=(double(q[tube.pivot])-p0)/(double(tube.last[tube.pivot])-p0);
    if(!std::isfinite(phase)||phase<0||phase>1) return false;
    for(int j=0;j<7;++j) {
        if(j==tube.pivot) continue;
        const double residual=double(q[j])-tube.first[j]-
            phase*(double(tube.last[j])-tube.first[j]);
        if(!std::isfinite(residual)||residual<tube.low[j]||residual>tube.high[j]) return false;
    }
    return true;
}

'''.replace("TUBE_ROWS",",\n".join(table))
    source=replace(source,marker,check+marker)
    source=replace(source,"acceptedCommonSnapshot.status.z<boxCount&&",
        "numiHumanRestingTerminalDomain(boxCount,acceptedCommonSnapshot.status.z,coordinates)&&")
    source=replace(source,"solved.status.z<boxCount&&finiteCoordinates&&",
        "numiHumanRestingTerminalDomain(boxCount,solved.status.z,coordinates)&&finiteCoordinates&&")
    # This file was linked into scratch before adding the host check.
    # Replace the scratch inode so the original checked source stays intact.
    temp=WORK/"NumiHumanRestingCoupling.hpp.tmp"
    temp.write_text(source)
    os.replace(temp,WORK/"NumiHumanRestingCoupling.hpp")



def patch_probe():
    name="NumiHumanRestingTransactionProbe.hpp"
    code=(SOURCE/name).read_text()
    start=code.index("    need(bleedReleaseStep!=NM_INVALID_INDEX")
    stop=code.index("\nconst char* fasciaRejectProbe",start)
    example=code[start:stop]
    example="\n".join("    "+line if line else line for line in example.splitlines())
    code=(code[:start]+'    if(bleedReleaseStep!=NM_INVALID_INDEX) {\n'+example+
          '\n    }\n'+code[stop:])
    target=WORK/(name+".tmp")
    target.write_text(code)
    os.replace(target,WORK/name)


def main():
    global WORK
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skin-trial",choices=("trimmed-guarded-pressure-skin","trimmed-oriented-pressure-skin"))
    args=parser.parse_args()
    if args.skin_trial:
        WORK=HERE/(args.skin_trial+"-build")
    WORK.mkdir(exist_ok=True)
    (WORK/"metalrobo").mkdir(exist_ok=True)
    original=(SHADER/"metalrobo/MetalRobo.metallib").resolve()
    dest=WORK/"metalrobo/MetalRobo.metallib"
    if not dest.is_file():
        os.link(original,dest)
    for p in ("NumiHumanAorticDiscretePreload.hpp","NumiHumanLocalBedLayers.hpp",
              "NumiHumanRestingCoupling.hpp","NumiHumanRestingTransactionProbe.hpp",
              "human_resting_runtime.hpp","numilab_human_myosim_visual_probe.mm","runtime.mm.o","build.py"):
        if not (WORK/p).is_file():
            os.link(SOURCE/p,WORK/p)
    if not (WORK/"include").is_dir():
        (WORK/"include").symlink_to(SOURCE/"include",target_is_directory=True)
    (WORK/"build_skin.py").write_bytes((SHADER/"build_skin.py").read_bytes())
    indices=bound_table()
    patch_metal()
    patch_host(args.skin_trial)
    patch_coupling()
    patch_probe()
    launched=dict(atlas_sha256=hash_file(ATLAS),proof_intervals=indices,
                  table_sha256=hash_file(WORK/"cardiac_proved_table.metalinc"),
                  source_anatomy_sha256=EXPECTED_GEOMETRY,source_field_sha256=EXPECTED_MAP)
    (WORK/"tube-contract.json").write_text(json.dumps(launched,indent=2)+"\n")
    for p in ("build_skin.py","build.py"):
        subprocess.run([sys.executable,str(WORK/p)],check=True)


if __name__=="__main__":
    main()
