"""Run the existing Apple-GPU resting physiology probe without anatomical claims."""
import argparse
import json
from pathlib import Path
import os
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
THUMB = HERE.parent / "coupled-thumb-reference-001"
OWNER = Path("/Users/numi/Work/skin-fat-native-integration-209")
BUILD = OWNER / "build-native"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps",type=int)
    args = parser.parse_args()
    deck=json.loads((THUMB/"combined-12s-no-export/launch.json").read_text())
    compile_commands=subprocess.check_output(
        ["/opt/homebrew/bin/ninja","-t","commands","numi-human-resting"],cwd=BUILD,text=True).splitlines()
    source=HERE/"human_resting_calibration.mm"
    program=HERE/"human-resting-calibration"
    obj=program.with_suffix(".o")
    if not program.is_file():
        original=(OWNER/"matter/tools/human_resting.mm").read_text()
        assert original.count("std::min(16u,steps-first)") == 1
        source.write_text(original.replace("std::min(16u,steps-first)","std::min(4u,steps-first)"))
        compile_argv=shlex.split(next(t for t in compile_commands if t.endswith("/human_resting.mm")))
        compile_argv[1:1]=["-I"+str(THUMB),"-I"+str(THUMB/"include"),"-I"+str(OWNER/"matter/tools")]
        compile_argv=[f'-DNUMI_HUMAN_RESPIRATION_METALLIB="{THUMB}/metalrobo/HumanRestingSkin.metallib"'
                      if t.startswith("-DNUMI_HUMAN_RESPIRATION_METALLIB=") else t for t in compile_argv]
        old_obj=compile_argv[compile_argv.index("-o")+1]
        compile_argv[-1]=str(source)
        compile_argv[compile_argv.index("-o")+1]=str(obj)
        compile_argv[compile_argv.index("-MT")+1]=str(obj)
        compile_argv[compile_argv.index("-MF")+1]=str(obj)+".d"
        last=compile_commands[-1]
        assert last.startswith(": && ") and last.endswith(" && :")
        link=shlex.split(last[5:-5])
        link[link.index("matter/libnumi_matter_runtime.a"):link.index("matter/libnumi_matter_runtime.a")]=[str(THUMB/"runtime.mm.o")]
        link[link.index("-o")+1]=str(program)
        link[link.index(old_obj)]=str(obj)
        commands=[compile_argv,link]
        (HERE/"calibration-build.json").write_text(json.dumps(commands,indent=2)+"\n")
        for command in commands:
            subprocess.run(command,cwd=BUILD,check=True)
    i=deck["argv"].index("--resting-scene")
    net,resp=deck["argv"][i+1:i+3]
    environment={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
    environment.update(deck["environment"])
    destination=HERE/f"physiology-dynamic-{args.steps}.csv"
    command=[str(program),net,resp,str(destination),"--steps",str(args.steps),"--dt","0.002","--vascular-dense45"]
    (HERE/f"physiology-dynamic-{args.steps}.json").write_text(json.dumps(
        {"command":command,"kind":"Stand-alone Brain–respiration–circulation calibration",
         "physical_body_or_viewer":False,"environment":deck["environment"]},indent=2)+"\n")
    subprocess.run(command,env=environment,cwd=BUILD,check=True)
if __name__=="__main__":
    main()
