"""Relink the patched statically owned Matter runtime into the native Human executable."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
prior = root.parent / "resting-clearance-and-atrium-007/shared-fascia-pressure-build"
app_baseline = root.parent / "resting-clearance-and-atrium-015/staged-solver-accuracy-build"
build = root / "pressure-with-abdominal-cavity-build"
directory = Path.home() / "Work/skin-fat-native-integration-209/build-native"
source = (prior / "runtime.mm").read_text()
before = """                    !world.vascular.cavities.empty() ||
                    !world.vascular.tissueBindings.empty() ||
                    world.dispatch.particleCount != 0u) {
                    diagnostics.message = "step external pressure requires a single lumped vascular network without continuum blood exchange";"""
after = """                    !world.vascular.tissueBindings.empty() ||
                    world.dispatch.particleCount != 0u) {
                    diagnostics.message = "step external pressure requires one vascular environment without continuum blood exchange";"""
assert source.count(before) == 1
(build / "app-runtime.mm").write_text(source.replace(before, after))
command = json.loads((prior / "build-commands.json").read_text())[0]
command = [str(build / ("app-" + Path(arg).name)) if Path(arg).parent == prior
           and Path(arg).name in {"runtime.mm", "runtime.mm.o"} else arg for arg in command]
assert str(build / "app-runtime.mm.o") in command and str(build / "app-runtime.mm") in command
subprocess.run(command, cwd=directory, check=True)
command = json.loads((app_baseline / "build-commands.json").read_text())[-1]
command = [str(build / "app-runtime.mm.o") if arg.endswith("/runtime.mm.o") else
           str(root / "floor-scene-build" / Path(arg).name) if Path(arg).parent == app_baseline else
           arg for arg in command]
subprocess.run(command, cwd=directory, check=True)
binary = root / "floor-scene-build/numi-human-native"
image = binary.read_bytes()
assert b"step external pressure requires a single lumped" not in image
assert b"step external pressure requires one vascular environment" in image
print("Native static Matter runtime boundary pin verified")
