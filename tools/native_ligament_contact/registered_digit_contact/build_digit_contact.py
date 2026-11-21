"""Extend the active Human owner and replay its retained Metal/ObjC++ build commands."""
import json,shutil,subprocess,sys,hashlib
from pathlib import Path
src,spec,out,app=map(Path,sys.argv[1:])
out.mkdir(parents=True,exist_ok=True);(out/'metalrobo').mkdir(exist_ok=True)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for f in src.iterdir():
 if f.is_file() and f.suffix in ['.hpp','.metalinc'] and not (out/f.name).exists():(out/f.name).hardlink_to(f)
# An experimental constitutive owner can accompany the Human overlay.
# Atomic replacement preserves the hard-linked control source.
for name in ('NumiHumanPelvicContact.hpp','pelvic_contact.metalinc'):
 if (spec/name).is_file():
  tmp=out/(name+'.replacement');shutil.copyfile(spec/name,tmp);tmp.replace(out/name)
lib=out/'metalrobo/MetalRobo.metallib'
if not lib.exists():lib.hardlink_to(src/'metalrobo/MetalRobo.metallib')
for n in ['NumiHumanDigitCorticalContact.hpp','registered_digit_contact.metalinc']:shutil.copyfile(spec/n,out/n)
(out/'human_respiration.metal').write_text((src/'human_respiration.metal').read_text()+'\n#include "registered_digit_contact.metalinc"\n')
s=(src/'numilab_human_myosim_visual_probe.mm').read_text()
def replace(a,b):
 global s
 if s.count(a)!=1:raise ValueError((a,s.count(a)))
 s=s.replace(a,b)
replace('struct MetalMujocoVisualQueries {','#ifdef NUMI_HUMAN_RESTING_SCENE\n#include "NumiHumanDigitCorticalContact.hpp"\n#endif\nstruct MetalMujocoVisualQueries {')
anchor='        if(supportGeometryProgram != nullptr && horizonInput.stand.enableContact) {'
replace(anchor,'''#ifdef NUMI_HUMAN_RESTING_SCENE
        NumiHumanDigitCorticalContact digitContact;
        const bool digitRows=NumiHumanDigitCorticalContact::requested();
        if(digitRows) {
            require(supportGeometryProgram && horizonInput.stand.enableContact && restingProgram,
                    "registered digit rows require the native resting contact transaction");
            digitContact.configure(*supportGeometryProgram,horizonInput,requestedSteps);
            horizonInput.stand.contacts=digitContact.contacts;
            horizonInput.points=digitContact.points;
            horizonInput.pointCount=digitContact.points.size();
        }
#endif
'''+anchor)
anchor='            horizonInput.stand.supportGeometryProgram=*supportGeometryProgram;'
replace(anchor,anchor+'''
#ifdef NUMI_HUMAN_RESTING_SCENE
            if(digitRows)horizonInput.stand.supportGeometryProgram=digitContact.program();
#endif''')
orig='''                (*acceptedObserver)(
                    completedSteps + segmentSteps, currentQ, currentV,
                    currentRoots.front(), segmentResult, queries);'''
replace(orig,'''                const auto observeExternal=[&]{
                    (*acceptedObserver)(completedSteps + segmentSteps,currentQ,currentV,
                        currentRoots.front(),segmentResult,queries);
                };
#ifdef NUMI_HUMAN_RESTING_SCENE
                if(digitRows){
                    digitContact.observe(segmentResult,completedSteps+segmentSteps);
                    digitContact.externalObservation(segmentResult,unsigned(model.articulations.front().nv),observeExternal);
                }else
#endif
                observeExternal();''')
(out/'numilab_human_myosim_visual_probe.mm').write_text(s)
cmd=[[arg.replace(str(src)+'/',str(out)+'/') for arg in row] for row in json.loads((src/'build-commands.json').read_text())]
(out/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
for row in cmd:subprocess.run(row,cwd=app/'build-native',check=True)
(out/'identities.json').write_text(json.dumps({str(p):sha(p) for p in [src/'numi-human-native',out/'numi-human-native',out/'metalrobo/HumanRestingSkin.metallib',out/'NumiHumanDigitCorticalContact.hpp',out/'registered_digit_contact.metalinc',spec/'registration.json']},indent=2)+'\n')
