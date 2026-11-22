"""Compile the new Metal mechanism against the verified parent source and native runtime."""
import json
from pathlib import Path
import subprocess
import sys
from material_context import sha
from runtime_patch import patch


def read(p):
    return json.loads(p.read_text())


def dump(path, value):
    path.write_text(json.dumps(value,indent=2)+"\n")


def build(root, library):
    case=root/"feasible-bed"
    prior=root/"normal-row"
    cert=root/"bed-certificate"
    out=case/"build"
    out.mkdir(exist_ok=True)
    if not (out/"src").exists():
        subprocess.run(["cp","-cR",str(cert/"build/src"),str(out/"src")],check=True)
    src=out/"src/metal/contact.metalinc"
    original=(cert/"build/src/metal/contact.metalinc").read_text()
    marker="} // namespace numi_matter_metal"
    if not original.rstrip().endswith(marker):
        raise ValueError("Contact namespace changed")
    src.unlink(missing_ok=True)
    src.write_text(original[:original.rindex(marker)]+(case/"contact.metalinc").read_text()+"\n"+marker+"\n")
    original=read(library/"identities.json")["commands"]
    cmds=[]
    def run(cmd):
        cmds.append(cmd)
        subprocess.run(cmd,check=True)
    cmd=read(root.parent/"velocity-consistent/identities.json")["commands"][0].copy()
    cmd[cmd.index("-c")+1]=str(out/"src/matter_kernels.metal")
    cmd[cmd.index("-o")+1]=str(out/"NumiMatter.air")
    run(cmd)
    run(["xcrun","-sdk","macosx","metallib",str(out/"NumiMatter.air"),"-o",str(out/"NumiMatter.metallib")])
    for i,name in [(2,"runtime.mm"),(3,"app-runtime.mm")]:
        code=patch((prior/"build"/name).read_text(),(case/"host.inc").read_text())
        (out/name).write_text(code)
        cmd=[str(out/name) if a==str(library/name) else str(out/(name+".o")) if a==str(library/(name+".o")) else a for a in original[i]]
        cmd[1:1]=["-iquote",str(library)]
        run(cmd)
    run([s.replace(str(library/"libmetalrobo.dylib"),str(out/"libmetalrobo.dylib")).replace(str(library/"runtime.mm.o"),str(out/"runtime.mm.o")) for s in original[4]])
    owner=case/"owner"
    if not owner.exists():
        subprocess.run(["cp","-cR",str(prior/"owner"),str(owner)],check=True)
    ident=read(prior/"owner/identities.json")
    cmd=[s.replace(str(prior/"owner")+"/",str(owner)+"/").replace(str(prior/"build")+"/",str(out)+"/") for s in ident["commands"][1]]
    cfg=read(cert/"parent/launch.json")
    subprocess.run(cmd,check=True,cwd=cfg["cwd"])
    exe=owner/"numi-human-native"
    ident.update(binary=sha(exe),binary_sha256=sha(exe),binary_path=str(exe))
    ident["commands"][1]=cmd
    dump(owner/"identities.json",ident)
    env=cfg["environment"]
    lib=out/"libmetalrobo.dylib"
    sh=out/"NumiMatter.metallib"
    del cfg["pins"][str(Path(env["DYLD_LIBRARY_PATH"])/lib.name)]
    del cfg["pins"][env["NUMI_MATTER_METALLIB_OVERRIDE"]]
    env.update(NM_FEM_REGISTERED_BED_DUAL="1",DYLD_LIBRARY_PATH=str(out),NUMI_MATTER_METALLIB_OVERRIDE=str(sh))
    cfg["pins"].update({str(lib):sha(lib),str(sh):sha(sh)})
    p=case/"parent"
    p.mkdir(exist_ok=True)
    dump(p/"launch.json",cfg)
    dump(out/"build-pins.json",dict(commands=cmds+[cmd],source={k:sha(case/k) for k in ["contact.metalinc","host.inc","runtime_patch.py","build.py"]},
        binary=sha(exe),dylib=sha(lib),metallib=sha(sh)))


if __name__=="__main__":
    build(*(Path(s).resolve() for s in sys.argv[1:]))
