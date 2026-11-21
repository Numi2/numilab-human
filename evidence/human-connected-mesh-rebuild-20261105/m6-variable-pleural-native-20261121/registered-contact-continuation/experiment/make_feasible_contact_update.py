"""Globalize the material displacement and its contact pressure using the existing cell limits."""
from pathlib import Path
import shutil,sys
src,dst=map(Path,sys.argv[1:]);dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/"pelvic_contact.metalinc";s=p.read_text()
at=s.index("kernel void nm_pelvic_schur_displacement")
end=s.index("struct PelvicSurfaceVertex",at);old=s[at:end]
t=old.replace("uint i [[thread_position_in_grid]]","device float4* proposal [[buffer(7)]],device atomic_uint* levels [[buffer(8)]],uint i [[thread_position_in_grid]]")
t=t.replace("displacement[i]=float4(target,0);","proposal[i]=float4(target,0);\n    if(i==m.x)atomic_store_explicit(levels+side,0u,memory_order_relaxed);")
assert t!=old and "proposal[i]=" in t
p.write_text(s[:at]+t+s[end:]+Path("pelvic_feasible_update.metalinc").read_text())
p=dst/"NumiHumanPelvicContact.hpp";s=p.read_text()
def before(needle,text):
    global s
    assert s.count(needle)==1,needle
    s=s.replace(needle,text+needle)
before("    std::map<unsigned,unsigned> spanIndex;", """    id<MTLBuffer> materialProposal=nil,materialStepLevels=nil;
    id<MTLComputePipelineState> materialStepBound=nil,materialStepApply=nil;
""")
before("        factorApply=pipeline(", """        materialProposal=alloc(countNodes*16);materialStepLevels=alloc(8);
        materialStepBound=pipeline(@"nm_pelvic_material_step_bound");
        materialStepApply=pipeline(@"nm_pelvic_material_step_apply");
""")
a="bind(6,pelvicHasSharedForward?lostSupport:needed);run((countNodes+127)/128,128);"
assert s.count(a)==1
s=s.replace(a,"bind(6,pelvicHasSharedForward?lostSupport:needed);bind(7,materialProposal);bind(8,materialStepLevels);run((countNodes+127)/128,128);")
a="            observe(1,1);bulkForces();select(1,true);nearest(1,1);"
before(a,"""            [e setComputePipelineState:materialStepBound];[e setBytes:&countCells length:4 atIndex:0];
            bind(1,cells);bind(2,refInverse);bind(3,totalDelta);bind(4,coordinates);bind(5,materialProposal);
            bind(6,needed);bind(7,materialStepLevels);run(countGroups,128);
            [e setComputePipelineState:materialStepApply];[e setBytes:&countNodes length:4 atIndex:0];
            bind(1,coordinates);bind(2,materialProposal);bind(3,pressures);bind(4,priorPressure);
            bind(5,materialStepLevels);bind(6,needed);run((std::max(512u,countNodes)+127)/128,128);
""")
p.write_text(s)
