from pathlib import Path
import shutil
src=Path('motion-registry-owner');dst=Path('material-search-owner');dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
with (dst/'pelvic_contact.metalinc').open('a') as f:f.write('\n'+Path('pelvic_material_search.metalinc').read_text())
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
def one(a,b):
    global s
    assert s.count(a)==1,a
    s=s.replace(a,b)
one('fitTractions=nil,residualDisplacement=nil','trialPotential=nil,choosePotential=nil,applyPotential=nil,fitTractions=nil,residualDisplacement=nil')
one('std::map<unsigned,unsigned> spanIndex;', 'id<MTLBuffer> trialTiles=nil,trialScale=nil;\n    std::map<unsigned,unsigned> spanIndex;')
one('motionOwners=load(45);','trialTiles=alloc(2u*countGroups*8u*sizeof(float));trialScale=alloc(2*sizeof(float));\n            trialPotential=pipeline(@"nm_pelvic_trial_potential");choosePotential=pipeline(@"nm_pelvic_choose_material_step");applyPotential=pipeline(@"nm_pelvic_apply_material_step");\n            motionOwners=load(45);')
one('std::vector<unsigned> sizes(gridSizes.begin()', 'gridSizes.insert(2u*countGroups);\n        std::vector<unsigned> sizes(gridSizes.begin()')
one('                correction(1);','''                if constexpr(pelvicMotionRegistry){
                    [e setComputePipelineState:trialPotential];words(0,{countCells,countNodes,countGroups,0});
                    bind(1,materialNeeded);bind(2,cells);bind(3,refInverse);bind(4,volumes);bind(5,totalDelta);
                    bind(6,baseShift);bind(7,tractionResidual);bind(8,trialTiles);run(2*countGroups,128);
                    [e setComputePipelineState:choosePotential];[e setBytes:&countGroups length:4 atIndex:0];
                    bind(1,materialNeeded);bind(2,trialTiles);bind(3,trialScale);run(2,32);
                    [e setComputePipelineState:applyPotential];[e setBytes:&countNodes length:4 atIndex:0];
                    bind(1,materialNeeded);bind(2,baseShift);bind(3,coordinates);bind(4,trialScale);run((countNodes+127)/128,128);
                }else correction(1);''')
p.write_text(s)
