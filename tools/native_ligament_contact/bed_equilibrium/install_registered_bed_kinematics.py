"""Build a matched native contact candidate on the existing full-body material solver."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from emit_skin_source_registration import sha

src, app, out = (Path(k).resolve() for k in sys.argv[1:])
pins = json.loads((src/'identities.json').read_text())
if pins[str(src/'numi-human-native')] != sha(src/'numi-human-native'):
    raise ValueError('unmatched native parent')
out.mkdir(exist_ok=False)
for item in src.iterdir():
    if item.is_file() and item.suffix in ('.hpp','.metal','.metalinc','.mm'):
        (out/item.name).hardlink_to(item)
(out/'metalrobo').symlink_to((src/'metalrobo').resolve(),target_is_directory=True)
name = 'NumiHumanRestingVisual.hpp'
reader = Path(__file__).with_name('NumiHumanRegisteredBedKinematics.inc')
target = (src/name).read_text()
marker = '        supportIdentity.integer(coupled.brain.rootProgramIdentity);'
assert target.count(marker)==1
target = target.replace(marker, '#include "NumiHumanRegisteredBedKinematics.inc"\n'+marker)
(out/name).unlink()
(out/name).write_text(target)
shutil.copyfile(reader,out/reader.name)
main='numilab_human_myosim_visual_probe.mm'
main_text=(src/main).read_text()
def change(a,b):
    global main_text
    if main_text.count(a)!=1:raise ValueError('unexpected native placement owner: '+a[:65])
    main_text=main_text.replace(a,b)
old='!requestedRootPose.has_value() && requestedPoseCoordinates.empty()),\n                    "--resting-initial-posture requires'
new='(!requestedRootPose.has_value() || (registeredBedRootPlacement && localBedLayersRequested() && !femSkinSupportRequested())) && requestedPoseCoordinates.empty()),\n                    "--resting-initial-posture requires'
change('            require(restingInitialPosture.empty() ||\n                        (restingScene.has_value()',
       '            const char* bedPlacementSetting=std::getenv("NUMI_HUMAN_LOCAL_BED_REGISTERED_OUTER");\n            const bool registeredBedRootPlacement=bedPlacementSetting && std::string_view(bedPlacementSetting)=="1";\n            require(restingInitialPosture.empty() ||\n                        (restingScene.has_value()')
change(old,new)
# An imported posture is already source-bounded and joint-equality projected.
# Apply an optional global root placement to that posture, not to the unrelated neutral shape.
change('            const bool explicitPoseOverride = !requestedPoseCoordinates.empty() ||\n                requestedRootPose.has_value();',
       '            const bool explicitPoseOverride = !requestedPoseCoordinates.empty() ||\n                (requestedRootPose.has_value() && restingInitialPosture.empty());')
change('                overriddenPoseQ = posture.q;\n                poseQ = overriddenPoseQ;',
       '                overriddenPoseQ = posture.q;\n                if (requestedRootPose) {\n                    require(registeredBedRootPlacement && overriddenPoseQ.size()>=7, "registered bed placement lost its donor");\n                    std::copy(requestedRootPose->begin(),requestedRootPose->end(),overriddenPoseQ.begin());\n                    std::cout<<"resting_registered_bed_root_placement=source_world_rigid_transform joint_coordinates_changed=0 applied_force_N=0"<<std::endl;\n                }\n                poseQ = overriddenPoseQ;')
(out/main).unlink()
(out/main).write_text(main_text)
cmds = [[arg.replace(str(src)+'/',str(out)+'/') for arg in cmd]
        for cmd in json.loads((src/'build-commands.json').read_text())[-2:]]
(out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
for cmd in cmds:
    subprocess.run(cmd,cwd=app/'build-native',check=True)
(out/'identities.json').write_text(json.dumps({str(k):sha(k) for k in (
    src/'numi-human-native',src/name,out/name,reader,out/main,out/'numi-human-native',
    out/'metalrobo/HumanRestingSkin.metallib')},indent=2)+'\n')
