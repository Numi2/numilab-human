"""Use the registered fine O bed support for an exact physical-volume release."""
from pathlib import Path
import json,sys,shutil,subprocess
from emit_skin_source_registration import sha
src,out,app=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
for p in src.iterdir():
 if p.is_file() and p.suffix in ('.hpp','.metalinc','.metal') and not (out/p.name).exists():(out/p.name).hardlink_to(p)
(out/'metalrobo').mkdir(exist_ok=True)
for p in (src/'metalrobo').glob('*.metallib'):
 if not (out/'metalrobo'/p.name).exists():(out/'metalrobo'/p.name).hardlink_to(p)
hpp=Path('NumiHumanRegisteredVolumeFirstTouch.hpp');shutil.copyfile(hpp,out/hpp.name)
n='numilab_human_myosim_visual_probe.mm';s=(src/n).read_text();s=s.replace('#include "NumiHumanSelectedSkinReference.hpp"', '#include "NumiHumanSelectedSkinReference.hpp"\n#include "NumiHumanRegisteredVolumeFirstTouch.hpp"')
a=s.index('                    require(makeWholeBodyFEMGroundAlignedSupport(poseQ,articulation.qOffset')
b=s.index('                    aligned.q.assign(',a)
old=s[a:b]
s=s[:a]+'''                    if(payloadSHA==selected_skin_reference::payload) {
                        std::vector<FEMOuterPointF32> sourceSeed;
                        const bool mapped=evaluateOuter(poseQ,sourceSeed,alignmentError);
                        require(mapped,"selected source registration failed: "+alignmentError);
                        const bool placed=checkRegisteredPhysicalOuterAtSeed(poseQ,header,physicalAlignment,alignmentError);
                        require(placed,"registered-volume first touch rejected: "+alignmentError);
                        std::cout<<"registered_volume_first_touch=passed native_F32_O_nodes="
                            <<physicalAlignment.outerNodeCount<<" minimum_gap_m="<<physicalAlignment.alignedMinimumGapM
                            <<" source_registration_rows="<<sourceSeed.size()
                            <<" root_changed=0 requires_exact_native_donor_preflight=1"<<std::endl;
                    } else {
'''+old+'                    }\n'+s[b:]
(out/n).write_text(s);cmd=[[a.replace(str(src)+'/',str(out)+'/') for a in row] for row in json.loads((src/'build-commands.json').read_text())]
(out/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
for row in cmd:subprocess.run(row,cwd=app/'build-native',check=True)
(out/'identity.json').write_text(json.dumps({str(p):sha(p) for p in [src/n,out/n,out/hpp.name,out/'NumiHumanSelectedSkinReference.hpp',out/'numi-human-native',*(out/'metalrobo').glob('*.metallib')]},indent=2)+'\n')
