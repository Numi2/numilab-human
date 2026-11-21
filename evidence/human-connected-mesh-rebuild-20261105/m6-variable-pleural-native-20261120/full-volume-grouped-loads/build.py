"""Apply the published grouped-force change to the exact full-volume runtime."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

here = Path(__file__).resolve().parent
extended = here.parent
scene = extended.parent
work = scene.parent
guarded = scene / "guarded-runtime-admission-build"
pressure = scene / "pressure-with-abdominal-cavity-build"
project = Path.home() / "Work/skin-fat-native-integration-209"
build_cwd = project / "build-native"
patch = extended / "grouped-contact-force-dispatch-028"
source = here / "candidate"
matter = source / "matter"
staged = matter / "src"
metal = staged / "metal"
include = matter / "include/numi/matter"
control = here / "control"

for folder in (metal, include):
    folder.mkdir(parents=True, exist_ok=True)
shutil.copyfile(pressure / "numi_human.mm", staged / "numi_human.mm")
shutil.copyfile(
    project / "matter/include/numi/matter/fem_body_contact_sample_grouping.hpp",
    include / "fem_body_contact_sample_grouping.hpp")
for path in (extended / "src").iterdir():
    if path.is_file() and path.name != "numi_human.mm":
        (staged / path.name).symlink_to(path)
for path in (extended / "src/metal").iterdir():
    if path.is_file() and path.name != "numi_human.metalinc":
        (metal / path.name).symlink_to(path)
shutil.copyfile(extended / "src/metal/numi_human.metalinc", metal / "numi_human.metalinc")
edit = (patch / "edit_dispatch.py").read_text()
old = 'text = replace(text, """        if (!(state_->femBodyContactSolverSamplesCount() == 0u) &&'
new = 'text = replace(text, """        if (!refreshGlides && state_->femBodyContactSolverSamplesCount() != 0u &&'
assert edit.count(old) == 1
edit = edit.replace(old, new).replace(
    '        if (state_->femBodyContactSolverSamplesCount() != 0u &&\n            !encodeKernel("""',
    '        if (!refreshGlides && state_->femBodyContactSolverSamplesCount() != 0u &&\n            !encodeKernel("""')
(here / "apply_grouped_force.py").write_text(edit)
subprocess.run(["python3", str(here / "apply_grouped_force.py")], cwd=here, check=True)

existing = json.loads((pressure / "build-commands.json").read_text())
host = here / "libmetalrobo.dylib"
adapter = here / "numi_human.mm.o"
host_commands = [
    [existing[1][0], "-I" + str(matter / "include"),
     *(str(staged / "numi_human.mm") if arg == str(pressure / "numi_human.mm")
       else str(adapter) if arg == str(pressure / "numi_human.mm.o")
       else arg for arg in existing[1][1:])],
    [str(host) if arg == str(pressure / "libmetalrobo.dylib")
     else str(adapter) if arg == str(pressure / "numi_human.mm.o")
     else str(guarded / "runtime.mm.o") if arg == str(pressure / "runtime.mm.o")
     else arg for arg in existing[-1]],
]
shader_commands = [
    ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-fno-fast-math",
     "-I", str(project / "matter/include"), "-I", str(project / "include"),
     "-c", str(staged / "matter_kernels.metal"), "-o", str(here / "NumiMatter.air")],
    ["xcrun", "-sdk", "macosx", "metallib", str(here / "NumiMatter.air"),
     "-o", str(here / "NumiMatter.metallib")],
]
all_commands = host_commands + shader_commands
(here / "build-commands.json").write_text(json.dumps(all_commands, indent=2) + "\n")
for command in all_commands:
    subprocess.run(command, cwd=build_cwd, check=True)
for path in (host, here / "NumiMatter.metallib"):
    with path.open("rb") as f:
        print(f"{hashlib.file_digest(f, 'sha256').hexdigest()}  {path}")
