"""Bind the registered opening frame to its installed physical material, before pack creation."""
from pathlib import Path
import json,shutil,subprocess,sys
from evaluate_registered_bed_transfer import digest
from build_distinct_registry import replace

def build(src,out,app):
    code=Path(__file__).resolve().parent
    binary="numi-human-native"; main="numilab_human_myosim_visual_probe.mm"
    info=json.loads((src/"identities.json").read_text())
    if digest(src/main)!=info["source"] or digest(src/binary)!=info["binary_sha256"]:
        raise ValueError("Changed native parent")
    out.mkdir(exist_ok=False)
    for f in src.iterdir():
        if f.is_file() and f.suffix in (".inc",".hpp",".metal",".metalinc",".metallib"):
            (out/f.name).hardlink_to(f)
    (out/"metalrobo").symlink_to((src/"metalrobo").resolve(),target_is_directory=True)
    h="NumiHumanInitialMaterialSurface.hpp";shutil.copyfile(code/h,out/h)
    txt=replace((src/main).read_text(),'#include "NumiHumanLoadedSkinPose.hpp"','#include "NumiHumanLoadedSkinPose.hpp"\n#include "'+h+'"')
    key="            numiLoadedSkin::install(installedObject,bundle,sidecars);"
    txt=replace(txt,key,key+'''
            if(numiLoadedSkin::enabled() && !wholeBodySkinRegisteredOuterTetrahedral &&
               wholeBodySkinRegisteredRenderPayload) {
                numiInitialMaterialSurface::bind(installedObject, wholeBodySkinRegisteredOuterVolumeNodeRows,
                                                 *wholeBodySkinRegisteredRenderPayload);
            }''')
    (out/main).write_text(txt)
    cmds=[[a.replace(str(src)+"/",str(out)+"/") for a in cmd] for cmd in info["commands"]]
    for cmd in cmds:subprocess.run(cmd,cwd=app/"build-native",check=True)
    info.update(commands=cmds,parent=info["binary_sha256"],binary=digest(out/binary),binary_sha256=digest(out/binary),
                source=digest(out/main),release_surface_header=digest(out/h))
    (out/"identities.json").write_text(json.dumps(info,indent=2)+"\n")

if __name__=="__main__":build(*(Path(v).resolve() for v in sys.argv[1:]))
