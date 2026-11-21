"""Keep the original pipeline and buffer contract for scenes without a motion registry."""
import shutil
from pathlib import Path
src=Path("facet-retry-owner");dst=Path("registered-continuation-owner");dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/"pelvic_contact.metalinc";s=p.read_text()
old=(Path("nodal-equilibrium-owner")/p.name).read_text()
begin="kernel void nm_pelvic_schur_displacement"
end="struct PelvicSurfaceVertex"
original=old[old.index(begin):old.index(end)]
modified=s[s.index(begin):s.index(end)]
key="    uint4 m=pelvicFlex[side];"
a=original.index(key);b=original.index("    displacement[i]=float4(target,0);")
calc=original[a:b]
helper="""inline float4 pelvicSchurDisplacementTarget(uint4 d, uint i, uint side, device const float* loads,
    device const float4* response, device const float4* displacement, device const float4* base,
    device const uint* lostSupport) {
"""+calc+"    return float4(target,0);\n}\n"
call="pelvicSchurDisplacementTarget(d,i,side,loads,response,displacement,base,lostSupport)"
first=original[:a]+"    displacement[i]="+call+";\n}\n"
a=modified.index(key)
second=modified[:a].replace("nm_pelvic_schur_displacement","nm_pelvic_propose_material_update")+"    proposal[i]="+call+";\n    if(i==pelvicFlex[side].x)atomic_store_explicit(levels+side,0u,memory_order_relaxed);\n}\n"
p.write_text(s.replace(modified,helper+first+second))
p=dst/"NumiHumanPelvicContact.hpp";s=p.read_text()
a=s.index("        materialProposal=alloc(");b=s.index("        factorApply=pipeline(",a);s=s[:a]+"        if constexpr(pelvicMotionRegistry){\n"+"\n".join("    "+v for v in s[a:b].splitlines())+"\n        }\n"+s[b:]
s=s.replace('moveNodes=pipeline(@"nm_pelvic_schur_displacement");','moveNodes=pipeline(pelvicMotionRegistry?@"nm_pelvic_propose_material_update":@"nm_pelvic_schur_displacement");')
s=s.replace('bind(7,materialProposal);bind(8,materialStepLevels);run(', 'if constexpr(pelvicMotionRegistry){bind(7,materialProposal);bind(8,materialStepLevels);}run(')
a=s.index("            [e setComputePipelineState:materialStepBound]");b=s.index("            observe(1,1);bulkForces();select(1,true);nearest(1,1);",a)
s=s[:a]+"            if constexpr(pelvicMotionRegistry){\n"+"\n".join("    "+v for v in s[a:b].splitlines())+"\n            }\n"+s[b:]
a=s.index("        auto solveEndpoint=[&](bool retry){\n")+len("        auto solveEndpoint=[&](bool retry){\n")
b=s.index("\n        };\n        solveEndpoint(false);",a)
s=s[:a]+"\n".join("    "+v if v.strip() else "" for v in s[a:b].splitlines())+s[b:]
p.write_text(s)
