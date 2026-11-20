"""Retain the unresolved DOF forces at strict supported initialization."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import subprocess

root = Path.cwd()
base = root / "staged-solver-accuracy-build"
build = root / "equilibrium-witness-build"
build.mkdir(exist_ok=True)
name = "numilab_human_myosim_visual_probe.mm"
before = (base / name).read_text()
key = "    std::string positionLimitDetails;\n"
assert before.count(key) == 1
diagnostic = r'''    if (requireBalanced && diagnostics.succeeded() && !diagnostics.balanced) {
        const auto& a = model.articulations.front();
        for (std::size_t v = 0u;
             v < compiled.generalizedAccelerationResidual.size(); ++v) {
            const auto& dof = model.dofs[a.vOffset + v];
            std::cout << std::setprecision(17)
                << "strict_equilibrium_residual"
                << " v_index=" << v
                << " q_index=" << dof.qIndex
                << " acceleration=" << compiled.generalizedAccelerationResidual.at(v)
                << " force=" << compiled.generalizedForceResidual.at(v)
                << " muscle_force=" << compiled.generalizedMuscleForce.at(v)
                << " support_force=" << compiled.generalizedSupportForce.at(v)
                << '\n';
        }
        std::cout << "strict_equilibrium_peak"
            << " acceleration_dof=" << diagnostics.maximumAccelerationResidualDof
            << " normalized_acceleration_dof=" << diagnostics.maximumNormalizedResidualDof
            << " active_joint_stop_rows=" << diagnostics.activePositionLimitCount
            << " activation_polish_steps=" << diagnostics.acceptedGlobalActivationPolishSteps
            << std::endl;
    }
'''
source = before.replace(key, diagnostic + key)
(build / name).write_text(source)
(root / "strict-equilibrium-witness.patch").write_text("".join(difflib.unified_diff(
    before.splitlines(keepends=True), source.splitlines(keepends=True),
    "a/apps/" + name, "b/apps/" + name,
)))
for path in (*base.glob("*.hpp"), base / "runtime.mm.o"):
    if (build / path.name).exists():
        assert os.path.samefile(path, build / path.name)
    else:
        os.link(path, build / path.name)
if not (build / "metalrobo").exists():
    (build / "metalrobo").symlink_to(base / "metalrobo", target_is_directory=True)
commands = [
    [arg.replace(str(base) + "/", str(build) + "/") for arg in command]
    for command in json.loads((base / "build-commands.json").read_text())
]
(build / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
with (root / "equilibrium-witness-build.log").open("wb") as log:
    for command in commands:
        subprocess.run(
            command, cwd=Path.home() / "Work/skin-fat-native-integration-209/build-native",
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
with (build / "numi-human-native").open("rb") as stream:
    print(hashlib.file_digest(stream, "sha256").hexdigest())
