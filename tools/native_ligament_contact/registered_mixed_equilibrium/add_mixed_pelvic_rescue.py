"""Add the coupled residual/Schur direction only after the unmodified solver rejects."""
from pathlib import Path
import sys
src,dst=map(Path,sys.argv[1:])
p=dst/'pelvic_contact.metalinc';p.write_text((src/p.name).read_text()+'\n'+Path('pelvic_mixed_rescue.metalinc').read_text())
p=dst/'NumiHumanPelvicContact.hpp';s=(src/p.name).read_text()
def one(a,b):
    global s
    if s.count(a)!=1:raise ValueError(a)
    s=s.replace(a,b)
one('id<MTLComputePipelineState> retrySnapshot=nil;', 'id<MTLComputePipelineState> retrySnapshot=nil,rescueResidual=nil,rescueRhs=nil,rescueProposal=nil;')
one('retrySnapshot=pipeline(@"nm_pelvic_retry_snapshot");','''retrySnapshot=pipeline(@"nm_pelvic_retry_snapshot");
            rescueResidual=pipeline(@"nm_pelvic_rescue_residual");
            rescueRhs=pipeline(@"nm_pelvic_rescue_rhs");
            rescueProposal=pipeline(@"nm_pelvic_rescue_proposal");''')
one('                solveEndpoint(true);','''                solveEndpoint(true);
                // Force and release gaps of one side can fail together. A single
                // coupled increment allows loaded contacts to release while the
                // free material coordinates move toward equilibrium.
                for(unsigned attempt=0;attempt<8;++attempt){
                    [e setComputePipelineState:makeSpans];[e setBytes:&count length:4 atIndex:0];
                    bind(1,needed);bind(2,dispatchSizes);bind(3,dispatchArgs);run((count+31)/32,32);optional=true;
                    [e setComputePipelineState:rescueResidual];[e setBytes:&countNodes length:4 atIndex:0];
                    bind(1,needed);bind(2,kr);bind(3,bulkGrad);bind(4,inc);bind(5,poolSlots);bind(6,pool);
                    bind(7,localN);bind(8,pressures);bind(9,tractionResidual);bind(10,baseShift);run((countNodes+127)/128,128);
                    [e setComputePipelineState:factorApply];bind(0,fRows);bind(1,fCols);bind(2,fVals);bind(3,fLevels);
                    bind(4,fOrder);bind(5,fPerm);bind(6,fScale);bind(7,tractionResidual);bind(8,baseShift);
                    bind(9,needed);bind(10,projRows);bind(11,projCols);bind(12,projVals);run(2,128);
                    [e setComputePipelineState:rescueRhs];bind(0,needed);bind(1,poolSlots);bind(2,pool);bind(3,localN);
                    bind(4,baseShift);bind(5,gram);bind(6,pressures);bind(7,rows);run(4,128);
                    [e setComputePipelineState:solve];words(0,{0,1,0,0});bind(1,poolSlots);bind(2,rows);bind(3,gram);
                    bind(4,pressures);bind(5,stats);bind(6,workspace);bind(7,needed);bind(8,priorPressure);run(2,32);
                    [e setComputePipelineState:rescueProposal];[e setBytes:&countNodes length:4 atIndex:0];
                    bind(1,needed);bind(2,coordinates);bind(3,baseShift);bind(4,columns);bind(5,priorPressure);
                    bind(6,pressures);bind(7,materialProposal);bind(8,materialStepLevels);run((countNodes+127)/128,128);
                    [e setComputePipelineState:materialStepBound];[e setBytes:&countCells length:4 atIndex:0];
                    bind(1,cells);bind(2,refInverse);bind(3,totalDelta);bind(4,coordinates);bind(5,materialProposal);
                    bind(6,needed);bind(7,materialStepLevels);run(countGroups,128);
                    [e setComputePipelineState:materialStepApply];[e setBytes:&countNodes length:4 atIndex:0];
                    bind(1,coordinates);bind(2,materialProposal);bind(3,pressures);bind(4,priorPressure);
                    bind(5,materialStepLevels);bind(6,needed);run((std::max(512u,countNodes)+127)/128,128);
                    observe(1,1);bulkForces();select(1,true);nearest(0,1);compliance();
                    [e setComputePipelineState:candidatePressure];words(0,{1,0,0,0});bind(1,poolSlots);
                    bind(2,rows);bind(3,gram);bind(4,pressures);bind(5,stats);bind(6,needed);run(2,32);
                    certify(128+attempt);optional=false;
                }''')
p.write_text(s)
