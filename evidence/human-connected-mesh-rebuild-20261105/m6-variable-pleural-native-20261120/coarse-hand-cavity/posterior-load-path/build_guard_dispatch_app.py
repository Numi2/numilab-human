"""Bind the actual bounded Newton trial to the existing final certificate."""
from argparse import ArgumentParser
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

NEW = """
            // A shorter budget can publish only a candidate that reached
            // the standard domain latch. The outer transaction owns retries.
            constexpr std::uint32_t requireShortTrialNewtonLatch = 1u << 31u;
            NMMatterDispatchGPU certificateDispatch = frameDispatch;
            if (state.femDomainNonlinearLatchEnabled &&
                nonlinearIterationBudget < state.mixedSolverValue.nonlinearIterations.x) {
                certificateDispatch.flags |= requireShortTrialNewtonLatch;
            }
            const auto bindCertificateDispatch = [&]() {
                setDispatch();
                [encoder setBytes:&certificateDispatch
                             length:sizeof(certificateDispatch) atIndex:0u];
            };
"""


def build(app, runtime, recipes, out, cwd):
    out.mkdir()
    original = runtime.read_text()
    start = original.index('            dispatchGroups32(\n                state.mixedCertificateTiledEnabled ? "nm_mixed_certify_partials"')
    end = original.index("            if (!encodeFEMNewtonCertificateTrace())", start)
    section = original[start:end]
    assert section.count("                setDispatch();\n") == 2
    section = section.replace("setDispatch();", "bindCertificateDispatch();")
    variant = original[:start] + NEW + section + original[end:]
    patched_runtime = out / "app-runtime.mm"
    patched_runtime.write_text(variant)
    (out / "short-guard-app-runtime.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), variant.splitlines(keepends=True),
        "a/matter/src/runtime.mm", "b/matter/src/runtime.mm")))
    template = json.loads(recipes.read_text())[0]
    core = [str(out / Path(arg).name) if arg.endswith(("app-runtime.mm", "app-runtime.mm.o")) else arg
            for arg in template]
    compile_app, link = json.loads((app / "build-commands.json").read_text())
    compile_app = [str(out / Path(p).name) if p.endswith("numi-human-native.o") else p
                   for p in compile_app]
    link = [str(out / Path(p).name) if Path(p).name in
            ("app-runtime.mm.o", "numi-human-native.o", "numi-human-native") else p for p in link]
    commands = [core, compile_app, link]
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with (out / "build.log").open("xb") as f:
        for command in commands:
            subprocess.run(command, cwd=cwd, stdout=f, stderr=f, check=True, timeout=240)
    (out / "binary.sha256").write_text(
        hashlib.sha256((out / "numi-human-native").read_bytes()).hexdigest() + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("app", "runtime", "recipes", "out", "cwd"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    build(**vars(parser.parse_args()))
