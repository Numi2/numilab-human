"""Use the existing verified abdominal-cavity admission in the current Human owner."""
from pathlib import Path
import json,subprocess,sys
from emit_skin_source_registration import sha
source,material,reference,out,app=map(lambda s:Path(s).resolve(),sys.argv[1:])
out.mkdir(exist_ok=False)
name='human_resting_runtime.hpp'; active=(source/name).read_text();known=(reference/name).read_text()
def section(t,begin,end):
    if t.count(begin)!=1 or t.count(end)!=1:raise ValueError('Ambiguous pressure ownership')
    a=t.index(begin);b=t.index(end,a);return t[a:b]
for a,b in [('        if (enabled("NM_VASCULAR_STEP_EXTERNAL_PRESSURE")) {','            stageThoracicPressure=pipeline('),('            if (stageThoracicPressure) {','                [enc setComputePipelineState:stageThoracicPressure];')]:
    active=active.replace(section(active,a,b),section(known,a,b),1)
for required in ('world.vascular.cavities[0].identity.x == 9007u','world.vascular.cavities[0].identity.y == 6u','(thoracicPressureMask & (1u << cavity.identity.y)) == 0u','v.microtick!=0u'):
    if required not in active:raise ValueError('Required pressure ownership guard missing')
(out/name).write_text(active)
main='numilab_human_myosim_visual_probe.mm'
for f in source.iterdir():
    if f.name!=name and f.is_file() and (f.suffix in ('.hpp','.metal','.metalinc','.metallib') or f.name==main):(out/f.name).hardlink_to(f)
(out/'metalrobo').symlink_to(source/'metalrobo',target_is_directory=True)
commands=json.loads((source/'build-commands.json').read_text())
compile=[s.replace(str(source)+'/',str(out)+'/') for s in commands[0]]
link=json.loads((material/'material-lowering-proof.json').read_text())['commands'][-1]
for i,x in enumerate(link):
    if x==str(source/'numi-human-native.o'):link[i]=str(out/'numi-human-native.o')
link[link.index('-o')+1]=str(out/'numi-human-native')
assert str(out/'numi-human-native.o') in link and str(material/'app-runtime.mm.o') in link
for cmd in [compile,link]:subprocess.run(cmd,cwd=app/'build-native',check=True)
(out/'pleural-integration.json').write_text(json.dumps(dict(commands=[compile,link],source_sha256=sha(source/name),reference_sha256=sha(reference/name),compiled_header_sha256=sha(out/name),executable_sha256=sha(out/'numi-human-native'),respiration_sha256=sha(out/'metalrobo/HumanRestingSkin.metallib')),indent=2)+'\n')
