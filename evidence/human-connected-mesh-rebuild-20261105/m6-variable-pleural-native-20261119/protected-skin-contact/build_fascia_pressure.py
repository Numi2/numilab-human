"""Stage one accepted pleural boundary per frame for a lumped circuit with FEM subcycles."""
import difflib
import json
import os
import subprocess

from run_support_dependency_preflight import HERE, read, write, digest

base=HERE/"shared-fascia-mass-build"
pressure=HERE.parent/"thoracic-vascular-001"
build=HERE/"shared-fascia-pressure-build"
build.mkdir()
for p in base.glob("*.hpp"):
    if p.name != "human_resting_runtime.hpp":
        os.link(p,build/p.name)
app="numi-human-native"
unit="numilab_human_myosim_visual_probe.mm"
os.link(base/unit,build/unit)
(build/"metalrobo").mkdir()
for p in (base/"metalrobo").iterdir():
    os.link(p,build/"metalrobo"/p.name)
changed={}
def edit(path,old,new):
    text=changed.get(path,path.read_text())
    assert text.count(old)==1
    changed[path]=text.replace(old,new)

runtime=pressure/"runtime.mm"
extension=base/"human_resting_runtime.hpp"
edit(runtime,'''                if (environments != 1u || world.vascular.compartments.empty() ||
                    !world.vascular.cavities.empty() || world.dispatch.femNodeCount != 0u ||
                    world.dispatch.particleCount != 0u || world.dispatch.maximumRateExponent != 0u) {
                    diagnostics.message = "step external pressure needs a single lumped, single-rate vascular network";''','''                // frameBegin stages the accepted pleural sample once. All
                // microticks in this frame reuse that same trial buffer,
                // including retries that begin from the accepted checkpoint.
                if (environments != 1u || world.vascular.compartments.empty() ||
                    !world.vascular.cavities.empty() ||
                    !world.vascular.tissueBindings.empty() ||
                    world.dispatch.particleCount != 0u) {
                    diagnostics.message = "step external pressure requires a single lumped vascular network without continuum blood exchange";''')
edit(extension,'''            need(world.vascular.compartments.size()==21u && dispatch.environmentCount==1u &&
                 expectedMicrotickCount==1u && world.vascular.cavities.empty(),
                 "thoracic pressure is registered for the 21-compartment lumped resting network");''','''            need(world.vascular.compartments.size()==21u && dispatch.environmentCount==1u &&
                 world.vascular.cavities.empty() && world.vascular.tissueBindings.empty() &&
                 dispatch.particleCount==0u,
                 "thoracic pressure is registered for the 21-compartment lumped resting network");''')
edit(extension,'''                if (!v.vascularExternalPressureSource || !v.vascularExternalPressureTrial ||
                    v.compartmentCount!=21u || v.microtickCount!=1u) {''','''                // The owner stages the last accepted pressure at frame entry.
                // The circulation sees it through all internal FEM microticks.
                if (!v.vascularExternalPressureSource || !v.vascularExternalPressureTrial ||
                    v.compartmentCount!=21u || v.microtick!=0u) {''')
patch=""
for path,owner in ((runtime,"matter/src/"),(extension,"matter/tools/")):
    (build/path.name).write_text(changed[path])
    patch+="".join(difflib.unified_diff(path.read_text().splitlines(keepends=True),
               changed[path].splitlines(keepends=True),"a/"+owner+path.name,"b/"+owner+path.name))
(HERE/"shared-fascia-frame-pressure.patch").write_text(patch)
app_cmds=read(base/"build-commands.json")
matter_cmd=json.JSONDecoder().raw_decode((pressure/"matter-build-command.json").read_text())[0][0]
paths={str(base/f):str(build/f) for f in (unit,app,app+".o")}
paths.update({str(pressure/f):str(build/f) for f in ("runtime.mm","runtime.mm.o")})
for arg in app_cmds[-1]:
    if arg.endswith("/runtime.mm.o"):
        assert digest(pressure/"runtime.mm.o")==digest(type(base)(arg))
        paths[arg]=str(build/"runtime.mm.o")
commands=[[paths.get(a,a) for a in cmd] for cmd in (matter_cmd,*app_cmds)]
write(build/"build-commands.json",commands)
for cmd in commands:
    subprocess.run(cmd,cwd="/Users/numi/Work/skin-fat-native-integration-209/build-native",check=True)
print(digest(build/app),flush=True)
