"""Make the FEM trial check run the same validity predicate as final validation."""
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess

here = Path(__file__).resolve().parent
root = Path("/Users/numi/Work/skin-fat-native-integration-209")
metals = root / "matter"
out = here / "exact-trial-shader"
out.mkdir()
shaders = out / "src"
shaders.mkdir()
for path in (metals / "src").iterdir():
    target = shaders / path.name
    if path.name == "metal":
        target.mkdir()
        for inner in path.iterdir():
            os.symlink(inner, target / inner.name)
    elif not path.is_dir():
        os.symlink(path, target)
source = metals / "src/metal/fem.metalinc"
candidate = shaders / "metal" / source.name
candidate.unlink()
original = source.read_text()
marker = """        float3x3 stress = evaluateAlgorithmicStress(
            material, programs, instructions,
            environmentParameters +
                environment * dispatch.parameterCount,
            nullptr, preciseLocalF,
            femMaterialInput(object, tetrahedron, deformationRate),
            dt, material.bulk.y, stressValid, deformationLow);"""
updated = """        const float3x3 preciseLocalFdot =
            femMaterialInput(object, tetrahedron, deformationRate);
        float validityMargin = 1.0f;
        if (!evaluateMaterialValidity(
                material, programs, instructions,
                environmentParameters + environment * dispatch.parameterCount,
                nullptr, preciseLocalF, preciseLocalFdot, dt, material.bulk.y,
                validityMargin, deformationLow) || !(validityMargin > 0.0f))
            return false;
        float3x3 stress = evaluateAlgorithmicStress(
            material, programs, instructions,
            environmentParameters + environment * dispatch.parameterCount,
            nullptr, preciseLocalF, preciseLocalFdot,
            dt, material.bulk.y, stressValid, deformationLow);"""
assert original.count(marker) == 1
result = original.replace(marker, updated)
candidate.write_text(result)
(here / "exact-trial-validity.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True), result.splitlines(keepends=True),
    "a/matter/src/metal/fem.metalinc", "b/matter/src/metal/fem.metalinc")))
# First translation unit must also be a real file so quoted relative includes
# resolve to the isolated FEM overlay.
kernel = shaders / "matter_kernels.metal"
kernel.unlink()
shutil.copy2(metals / "src/matter_kernels.metal", kernel)
intermediate = Path("/Volumes/NumiHumanTrial/strict-pectoral.air")
library = out / "NumiMatter.metallib"
common = ["xcrun", "-sdk", "macosx"]
commands = [
    common + ["metal", "-std=metal4.0", "-O3", "-fno-fast-math",
              "-I", str(metals / "include"), "-I", str(root / "include"),
              "-c", str(kernel), "-o", str(intermediate)],
    common + ["metallib", str(intermediate), "-o", str(library)],
]
(out / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
for command in commands:
    subprocess.run(command, check=True)
intermediate.unlink()
