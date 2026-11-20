"""Build the unchanged and one-correction Metal variants with the production flags."""
from argparse import ArgumentParser
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def build(source, out):
    out.mkdir()
    shader_dir = source / "matter/src"
    name = "metal/fgmres.metalinc"
    original = (shader_dir / name).read_text()
    candidate = original
    marker = "// Compact begin finalizer preserves the nonlinear latch"
    start = original.index(marker)
    old = """    const bool correctionFinite = nonlinearIteration > 1u &&
        isfinite(previousNonlinear.z);"""
    new = """    // Iteration one already follows a measured Newton correction. Only the
    // initial guess lacks that history. For the independently certified domains,
    // let a fresh physical residual stop the next correction at either point.
    const uint correctionHistoryStart = (layout.reserved & 2u) != 0u ? 0u : 1u;
    const bool correctionFinite = nonlinearIteration > correctionHistoryStart &&
        isfinite(previousNonlinear.z);"""
    assert original[start:].count(old) == 1
    candidate = original[:start] + original[start:].replace(old, new, 1)
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True),
        candidate.splitlines(keepends=True), "a/matter/src/" + name, "b/matter/src/" + name))
    (out / "one-correction-domain-latch.patch").write_text(patch)
    selected = out / name
    selected.parent.mkdir()
    top = out / "matter_kernels.metal"
    top.write_bytes((shader_dir / top.name).read_bytes())
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [shader_dir / top.name, *sorted((shader_dir / "metal").glob("*.metalinc"))]}
    commands = []
    for variant, definition in (("control", original), ("one-correction", candidate)):
        selected.write_text(definition)
        air = out / "temporary.air"
        library = out / f"{variant}.metallib"
        chain = [
            ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-fno-fast-math",
             "-I", str(source / "matter/include"), "-I", str(source / "include"),
             "-I", str(shader_dir), "-I", str(shader_dir / "metal"), "-c", str(top), "-o", str(air)],
            ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)],
        ]
        commands.append({"variant": variant, "commands": chain})
        with (out / f"{variant}.log").open("xb") as f:
            for cmd in chain:
                subprocess.run(cmd, stdout=f, stderr=f, check=True, timeout=180)
        air.unlink()
        hashes[library.name] = hashlib.sha256(library.read_bytes()).hexdigest()
    (out / "build.json").write_text(json.dumps({"sha256": hashes, "variants": commands}, indent=2) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("source", "out"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    build(**vars(parser.parse_args()))
