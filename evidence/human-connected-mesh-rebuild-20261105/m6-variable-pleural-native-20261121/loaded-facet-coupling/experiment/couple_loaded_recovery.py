from pathlib import Path
import shutil
src=Path('active-facet-owner');dst=Path('mixed-recovery-owner');dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/'pelvic_contact.metalinc';s=p.read_text();old='q.x<=tolerance || q.z>=5.e-7f || q.w>2.e-9f';assert s.count(old)==1
s=s.replace(old,'q.x<=tolerance')+'''
// At fixed geometry an open unilateral row has zero compressive traction.
// The solver has to recover any resulting material imbalance before admission.
kernel void nm_pelvic_unload_open(device const uint* needed [[buffer(0)]],device const uint* slots [[buffer(1)]],
    device const PelvicRow* rows [[buffer(2)]],device float* load [[buffer(3)]],uint i [[thread_position_in_grid]]){
    if(i<512 && needed[i/256] && slots[i]!=UINT_MAX && rows[i].leverB.w>=5.e-7f)load[i]=0.f;
}
'''
p.write_text(s)
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
def one(a,b):
    global s
    assert s.count(a)==1,a
    s=s.replace(a,b)
one('nodeEquilibrium=nil;','nodeEquilibrium=nil,unloadOpen=nil;')
one('nodeEquilibrium=pipeline(@"nm_pelvic_nodal_equilibrium");','nodeEquilibrium=pipeline(@"nm_pelvic_nodal_equilibrium");unloadOpen=pipeline(@"nm_pelvic_unload_open");')
one('unsigned(outer==15u || outer==31u)','unsigned(outer==15u || outer==31u || (outer>=48u && outer!=UINT_MAX))')
key='                [e setComputePipelineState:nodeEquilibrium];'
one(key,'''                [e setComputePipelineState:unloadOpen];bind(0,needed);bind(1,poolSlots);bind(2,rows);bind(3,pressures);run(4,128);
                [e setComputePipelineState:clearInc];[e setBytes:&countNodes length:4 atIndex:0];bind(1,inc);run((countNodes*8+127)/128,128);
                [e setComputePipelineState:buildInc];bind(0,poolSlots);bind(1,pressures);bind(2,pool);bind(3,inc);run(16,128);
                [e setComputePipelineState:fitTractions];bind(0,needed);bind(1,poolSlots);bind(2,pool);bind(3,localN);
                bind(4,rows);bind(5,kr);bind(6,bulkGrad);bind(7,pressures);bind(8,inc);bind(9,tractionResidual);run(2,32);
                [e setComputePipelineState:clearInc];[e setBytes:&countNodes length:4 atIndex:0];bind(1,inc);run((countNodes*8+127)/128,128);
                [e setComputePipelineState:buildInc];bind(0,poolSlots);bind(1,pressures);bind(2,pool);bind(3,inc);run(16,128);
'''+key)
start=s.index(key);end=s.index('                certify(48+local);',start)
t=s[start:end]
old='observe(1,1);bulkForces();select(1,true);nearest(0,1);compliance();';assert t.count(old)==1
t=t.replace(old,'''observe(1,1);bulkForces();select(1,false);nearest(0,1);compliance();
                // Use an incremental contact solve for the refreshed material
                // surface and apply its displacement and pressure as one update.
                [e setComputePipelineState:shiftDual];bind(0,needed);bind(1,poolSlots);bind(2,pressures);bind(3,gram);bind(4,rows);run(4,128);
                [e setComputePipelineState:solve];words(0,{0,1,0,0});bind(1,poolSlots);bind(2,rows);bind(3,gram);bind(4,pressures);
                bind(5,stats);bind(6,workspace);bind(7,needed);bind(8,priorPressure);run(2,32);
                [e setComputePipelineState:dualMove];[e setBytes:&countNodes length:4 atIndex:0];bind(1,needed);bind(2,priorPressure);
                bind(3,pressures);bind(4,columns);bind(5,coordinates);bind(6,zeroMaterialPhase);run((countNodes+127)/128,128);
                [e setComputePipelineState:dualLoad];bind(0,needed);bind(1,priorPressure);bind(2,pressures);bind(3,zeroMaterialPhase);run(4,128);
                observe(1,1);bulkForces();select(1,true);nearest(0,1);compliance();''')
s=s[:start]+t+s[end:]
one('id<MTLBuffer> constructionMaterialGuess=nil,','id<MTLBuffer> zeroMaterialPhase=nil,constructionMaterialGuess=nil,')
one('            nodeEquilibrium=pipeline(', '            zeroMaterialPhase=alloc(8,true);std::memset(zeroMaterialPhase.contents,0,8);\n            nodeEquilibrium=pipeline(')
p.write_text(s)
