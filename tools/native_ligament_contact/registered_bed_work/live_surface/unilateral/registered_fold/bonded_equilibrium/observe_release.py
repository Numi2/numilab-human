"""Export the same GPU-evaluated initial body poses consumed by the native resting renderer."""
from pathlib import Path
import sys,json,subprocess
from material_context import sha

def with_pose_capture(old):
    anchor='                    const auto initialBodies=nativePose(poseQ),restBodies=nativePose(rigid.model.defaultQ);'
    if old.count(anchor)!=1:raise ValueError('Unrecognized executed initial pose owner')
    new=old.replace(anchor,anchor+'''
                    if(const char* snapshot=std::getenv("NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP")) {
                        const auto dst=std::filesystem::path(snapshot).parent_path()/"native-initial-bodies.json";
                        writeSkinPoseSnapshot(*skinPayload, initialBodies, restBodies, dst);
                    }''')
    return new

def build(base):
    source=base/'owner';out=base/'observed-owner'
    if not out.exists():subprocess.run(['cp','-cR',str(source),str(out)],check=True)
    data=json.loads((source/'identities.json').read_text())
    main='numilab_human_myosim_visual_probe.mm'; old=(source/main).read_text()
    new=with_pose_capture(old)
    (out/main).write_text(new)
    commands=[[s.replace(str(source)+'/',str(out)+'/') for s in c] for c in data['commands']]
    if str(out/main) not in commands[0]:raise ValueError('Source compiler does not address observed owner')
    launch_path=base/'three/launch.json'
    if not launch_path.exists():launch_path=base.parent/'free-interface/three/launch.json'
    launch=json.loads(launch_path.read_text())
    for cmd in commands:subprocess.run(cmd,cwd=launch['cwd'],check=True)
    binary=out/'numi-human-native'
    data.update(binary=sha(binary),binary_sha256=sha(binary),binary_path=str(binary),commands=commands,source=sha(out/main),observed_owner=sha(source/main))
    (out/'identities.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
