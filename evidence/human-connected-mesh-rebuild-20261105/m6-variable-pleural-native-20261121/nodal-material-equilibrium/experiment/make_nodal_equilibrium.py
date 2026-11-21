from pathlib import Path
import shutil
src=Path('verified-loaded-owner');dst=Path('nodal-equilibrium-owner');dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/'pelvic_contact.metalinc'
with p.open('a') as f:f.write('\n'+Path('pelvic_nodal_equilibrium.metalinc').read_text())
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
def one(a,b):
    global s
    assert s.count(a)==1,a
    s=s.replace(a,b)
one('seedPressures=nil;', 'seedPressures=nil,nodeEquilibrium=nil;')
one('            motionOwners=load(45);', '''            nodeEquilibrium=pipeline(@"nm_pelvic_nodal_equilibrium");
            motionOwners=load(45);''')
m='        [e setComputePipelineState:gate];words(0,{p.stepIndex,0,0,unsigned(post)});'
one(m,'''        if constexpr(pelvicMotionRegistry){
            for(unsigned local=0;local<4;++local){
                unsigned count=unsigned(spanIndex.size());
                [e setComputePipelineState:makeSpans];[e setBytes:&count length:4 atIndex:0];
                bind(1,needed);bind(2,dispatchSizes);bind(3,dispatchArgs);run((count+31)/32,32);optional=true;
                [e setComputePipelineState:nodeEquilibrium];bind(0,needed);bind(1,quality);bind(2,kr);bind(3,inc);
                bind(4,projRows);bind(5,bulkGrad);bind(6,totalDelta);bind(7,cellRows);bind(8,cellCorners);bind(9,cells);
                bind(10,refInverse);bind(11,volumes);bind(12,coordinates);run(2,32);
                observe(1,1);bulkForces();select(1,true);nearest(0,1);compliance();
                [e setComputePipelineState:candidatePressure];words(0,{1,0,0,0});bind(1,poolSlots);
                bind(2,rows);bind(3,gram);bind(4,pressures);bind(5,stats);bind(6,needed);run(2,32);
                certify(48+local);optional=false;
            }
        }
'''+m)
p.write_text(s)
