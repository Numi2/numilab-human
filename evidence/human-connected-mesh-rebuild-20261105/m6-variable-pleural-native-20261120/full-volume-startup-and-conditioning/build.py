"""Build the published private-buffer zero-fill change on the grouped Human host."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

here = Path(__file__).resolve().parent
guarded = here.parent.parent / "guarded-runtime-admission-build"
pressure = here.parent.parent / "pressure-with-abdominal-cavity-build"
grouped = here.parent / "full-volume-grouped-loads-031"
workspace = Path.home() / "Work/skin-fat-native-integration-209/build-native"
patch = here.parent / "zero-contact-history-initialization/zero-contact-history.patch"
source = here / "matter/src/runtime.mm"
source.parent.mkdir(parents=True)
shutil.copyfile(guarded / "runtime.mm", source)
subprocess.run(["patch", "--fuzz=0", "-p1", "-i", str(patch)], cwd=here, check=True)
guarded_object = guarded / "runtime.mm.o"
compile_object = here / guarded_object.name
link_target = here / "libmetalrobo.dylib"
base_compile = json.loads((pressure / "build-commands.json").read_text())[0]
base_link = json.loads((grouped / "build-commands.json").read_text())[1]
assert base_compile.count(str(pressure / "runtime.mm")) == 1
assert base_compile.count(str(pressure / "runtime.mm.o")) == 1
assert base_link.count(str(guarded_object)) == 1
compile_command = [str(source) if arg == str(pressure / "runtime.mm")
                   else str(compile_object) if arg == str(pressure / "runtime.mm.o")
                   else arg for arg in base_compile]
link_command = [str(link_target) if arg == str(grouped / link_target.name)
                else str(compile_object) if arg == str(guarded_object)
                else arg for arg in base_link]
commands = [compile_command, link_command]
(here / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
for command in commands:
    subprocess.run(command, cwd=workspace, check=True)
for path in (link_target, source, patch, grouped / "NumiMatter.metallib"):
    with path.open("rb") as stream:
        print(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {path}", flush=True)
