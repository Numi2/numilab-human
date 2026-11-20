"""Build a short-Newton trial that must reach the ordinary physical latch."""
from argparse import ArgumentParser
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

GUARD = """
    // A short trial may remove encoded Newton work only after the same
    // nonlinear latch used by the full budget has fired. Without it the
    // outer transaction must roll back before retrying from the predecessor.
    if (lane == 0u && objectIndex == telemetryWriterObject &&
        shortNewtonBudget != 0u &&
        fgmresStates[environment].nonlinear.w != 1.0f) {
        recordFailure(statuses[environment], NM_STATUS_NONLINEAR_SOLVER_FAILURE,
            objectIndex, NM_INVALID_INDEX,
            float4(relativeResidual, relativeCorrection, 0.0f,
                float(shortNewtonBudget)));
    }
"""


def build(source, out):
    out.mkdir()
    shader = source / "matter/src"
    name = "metal/mixed_fem.metalinc"
    original = (shader / name).read_text()
    end = "        minimumJ, volumeNorm2, activeElements);\n}"
    assert original.count(end) == 2
    variant = original.replace(end, end[:-1] + GUARD + "}")
    for kernel in ("nm_mixed_certify(", "nm_mixed_certify_finish("):
        start = variant.index("kernel void " + kernel)
        anchor = "    uint lane [[thread_index_in_threadgroup]],"
        index = variant.index(anchor, start)
        variant = variant[:index] + "    constant uint& shortNewtonBudget [[buffer(29)]],\n" + variant[index:]
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True),
        variant.splitlines(keepends=True), "a/matter/src/" + name, "b/matter/src/" + name))
    (out / "latch-required-short-trial.patch").write_text(patch)
    selected = out / name
    selected.parent.mkdir()
    selected.write_text(variant)
    top = out / "matter_kernels.metal"
    top.write_bytes((shader / top.name).read_bytes())
    air = out / "temporary.air"
    metal = out / "guarded-short.metallib"
    commands = [
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-fno-fast-math",
         "-I", str(source / "matter/include"), "-I", str(source / "include"),
         "-I", str(shader), "-I", str(shader / "metal"), "-c", str(top), "-o", str(air)],
        ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(metal)],
    ]
    with (out / "build.log").open("xb") as f:
        for command in commands:
            subprocess.run(command, stdout=f, stderr=f, check=True, timeout=180)
    air.unlink()
    (out / "build.json").write_text(json.dumps({
        "commands": commands, "parent_shader_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "metallib_sha256": hashlib.sha256(metal.read_bytes()).hexdigest()
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("source", "out"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    build(**vars(parser.parse_args()))
