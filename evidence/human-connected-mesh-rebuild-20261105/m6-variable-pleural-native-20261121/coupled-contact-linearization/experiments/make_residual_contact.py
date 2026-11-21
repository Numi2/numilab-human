from pathlib import Path
import shutil
src=Path('motion-registry-owner');dst=Path('residual-contact-owner');dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
shutil.copyfile('pelvic_residual_contact.metalinc',dst/'pelvic_residual_contact.metalinc')
p=dst/'build_pelvic_coupling.py';s=p.read_text().replace("'pelvic_motion_registry.metalinc'):","'pelvic_motion_registry.metalinc','pelvic_residual_contact.metalinc'):")
s=s.replace('#include "pelvic_motion_registry.metalinc"','#include "pelvic_motion_registry.metalinc"\\n#include "pelvic_residual_contact.metalinc"');p.write_text(s)
p=dst/'pelvic_end_geometry.metalinc';s=p.read_text().replace('bool test=!control[0]&&!needed[0]&&!needed[1];','bool test=!control[0]&&((!needed[0]&&!needed[1])||d.z);')
s=s.replace('if(r[0]==0)control[0]=1;', 'if(r[0]==0)control[0]=(!needed[0]&&!needed[1]);');p.write_text(s)
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
s=s.replace('motionBind=nil;', 'motionBind=nil,recoveryRhs=nil,recoveryPressure=nil,recoveryStep=nil;')
s=s.replace('motionOwners=load(45);','recoveryRhs=pipeline(@"nm_pelvic_residual_rhs");recoveryPressure=pipeline(@"nm_pelvic_incremental_compression");recoveryStep=pipeline(@"nm_pelvic_residual_step");motionOwners=load(45);')
s=s.replace('words(0,{nCrossingGroups,unsigned(outer==UINT_MAX),0,0})','words(0,{nCrossingGroups,unsigned(outer==UINT_MAX),unsigned(outer!=UINT_MAX&&outer>=48),0})')
mark='        [e setComputePipelineState:gate];words(0,{p.stepIndex,0,0,unsigned(post)});';assert s.count(mark)==1
s=s.replace(mark,'''        if constexpr(pelvicMotionRegistry){if(post){
            const float relaxation=.5f;
            for(unsigned sweep=0;sweep<12;++sweep){
                unsigned grids=unsigned(spanIndex.size());
                [e setComputePipelineState:makeSpans];[e setBytes:&grids length:4 atIndex:0];bind(1,needed);bind(2,dispatchSizes);bind(3,dispatchArgs);
                run((grids+31)/32,32);optional=true;
                observe(1,1);select(1,false);nearest(0,1);compliance();
                [e setComputePipelineState:clearInc];[e setBytes:&countNodes length:4 atIndex:0];bind(1,inc);run((countNodes*8+127)/128,128);
                [e setComputePipelineState:buildInc];bind(0,poolSlots);bind(1,pressures);bind(2,pool);bind(3,inc);run(16,128);
                [e setComputePipelineState:recoveryRhs];bind(0,kr);bind(1,bulkGrad);bind(2,inc);bind(3,poolSlots);bind(4,pressures);bind(5,localN);
                bind(6,pool);bind(7,tractionResidual);bind(8,baseShift);[e setBytes:&countNodes length:4 atIndex:9];run((countNodes+127)/128,128);
                [e setComputePipelineState:factorApply];bind(0,fRows);bind(1,fCols);bind(2,fVals);bind(3,fLevels);bind(4,fOrder);bind(5,fPerm);bind(6,fScale);
                bind(7,tractionResidual);bind(8,baseShift);bind(9,needed);bind(10,projRows);bind(11,projCols);bind(12,projVals);run(2,128);
                [e setComputePipelineState:recoveryPressure];bind(0,poolSlots);bind(1,rows);bind(2,gram);bind(3,needed);bind(4,pressures);bind(5,priorPressure);
                bind(6,baseShift);bind(7,localN);bind(8,pool);[e setBytes:&relaxation length:4 atIndex:9];run(2,256);
                [e setComputePipelineState:recoveryStep];[e setBytes:&countNodes length:4 atIndex:0];bind(1,needed);bind(2,coordinates);bind(3,baseShift);
                bind(4,columns);bind(5,priorPressure);[e setBytes:&relaxation length:4 atIndex:6];run((countNodes+127)/128,128);
                observe(1,1);bulkForces();select(1,true);nearest(0,1);compliance();
                [e setComputePipelineState:candidatePressure];words(0,{1,0,0,0});bind(1,poolSlots);bind(2,rows);bind(3,gram);
                bind(4,pressures);bind(5,stats);bind(6,needed);run(2,32);
                certify(48+sweep);optional=false;
            }
        }}
'''+mark)
p.write_text(s)
