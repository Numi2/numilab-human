"""Build original-terminal Newton staging in the two retained Matter owners."""
from argparse import ArgumentParser
import difflib
import json
from pathlib import Path
import subprocess
import sys

BUDGET = """            const std::uint32_t nonlinearIterationBudget =
                nonlinearBudgetOverride == 0u
                    ? state.mixedSolverValue.nonlinearIterations.x
                    : nonlinearBudgetOverride;"""
SCHEDULE = r"""            const std::uint32_t cookedNonlinearIterationBudget =
                state.mixedSolverValue.nonlinearIterations.x;
            const std::uint32_t scheduledNonlinearIterations =
                nonlinearBudgetOverride == 0u
                    ? cookedNonlinearIterationBudget : nonlinearBudgetOverride;
            const char* terminalSchedule = std::getenv("NUMI_MATTER_KEEP_ORIGINAL_NEWTON_TERMINAL");
            std::uint32_t referenceNonlinearIterationBudget = cookedNonlinearIterationBudget;
            if (terminalSchedule != nullptr) {
                // Reference cap includes an explicit runtime override, when used.
                char* end = nullptr;
                const auto reference = std::strtoul(terminalSchedule, &end, 10);
                if (end != terminalSchedule && *end == '\0' &&
                    reference >= 2u && reference <= 24u) {
                    referenceNonlinearIterationBudget = static_cast<std::uint32_t>(reference);
                }
            }
            const bool preserveTerminalSchedule =
                terminalSchedule != nullptr && state.femDomainNonlinearLatchEnabled &&
                scheduledNonlinearIterations >= 2u &&
                scheduledNonlinearIterations < referenceNonlinearIterationBudget;
            const std::uint32_t nonlinearIterationBudget =
                preserveTerminalSchedule
                    ? referenceNonlinearIterationBudget : scheduledNonlinearIterations;"""
LOOP = """            for (std::uint32_t nonlinearIteration = 0u;
                 nonlinearIteration < nonlinearIterationBudget;
                 ++nonlinearIteration) {"""
JUMP = """            for (std::uint32_t scheduled = 0u;
                 scheduled < scheduledNonlinearIterations; ++scheduled) {
            const bool jumpToTerminal = preserveTerminalSchedule &&
                scheduled + 1u == scheduledNonlinearIterations;
            const std::uint32_t nonlinearIteration =
                jumpToTerminal ? nonlinearIterationBudget - 1u : scheduled;
            if (jumpToTerminal) {
                dispatchThreads("nm_mixed_require_latch_before_terminal_newton",
                    environments, [&] {
                    setDispatch();
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:1u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:2u];
                });
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }"""
KERNEL = """
// The staged host may omit only iterations after the ordinary GPU latch.
// Check before the original terminal assembly, which still runs normally.
kernel void nm_mixed_require_latch_before_terminal_newton(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    device const NMFGMRESStateGPU* states [[buffer(1)]],
    device NMMatterStatusGPU* statuses [[buffer(2)]],
    uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        !environmentHealthy(statuses, environment)) return;
    const float4 nonlinear = states[environment].nonlinear;
    if (nonlinear.w != 1.0f)
        recordFailure(statuses[environment], NM_STATUS_NONLINEAR_SOLVER_FAILURE,
            NM_INVALID_INDEX, NM_INVALID_INDEX, nonlinear);
}
"""


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def runtime_variant(original):
    out = replace_once(original, BUDGET, SCHEDULE)
    out = replace_once(out, LOOP, JUMP)
    out = replace_once(out,
        '\n            "nm_mixed_certify",',
        '\n            "nm_mixed_require_latch_before_terminal_newton",\n            "nm_mixed_certify",')
    # The final certificate retains the existing short-trial latch check.
    return replace_once(out,
        '                nonlinearIterationBudget < state.mixedSolverValue.nonlinearIterations.x) {',
        '                (preserveTerminalSchedule || scheduledNonlinearIterations < cookedNonlinearIterationBudget)) {')


def patch_file(path, output, variant):
    old = path.read_text()
    new = variant(old)
    output.write_text(new)
    patch = "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        "a/" + path.name, "b/" + path.name))
    (output.parent / (output.name + ".patch")).write_text(patch)


def relocate(commands, original, work):
    return [[str(work / Path(part).name) if Path(part).parent == original else part
             for part in command] for command in commands]


def build(mode, previous, source, out):
    sys.path.insert(0, str(previous))
    from replay_outlier import digest

    out.mkdir()
    if mode == "shader":
        owner = previous / "guarded-dispatch-shader"
        (out / "metal").mkdir()
        patch_file(owner / "metal/mixed_fem.metalinc", out / "metal/mixed_fem.metalinc",
                   lambda old: replace_once(old, "} // namespace numi_matter_metal",
                                            KERNEL + "\n} // namespace numi_matter_metal"))
        (out / "matter_kernels.metal").write_bytes((owner / "matter_kernels.metal").read_bytes())
        commands = relocate(json.loads((owner / "build.json").read_text())["commands"], owner, out)
        final = out / "guarded-short.metallib"
        parents = [owner / "metal/mixed_fem.metalinc"]
    elif mode == "library":
        owner = previous / "guarded-dispatch-library"
        patch_file(owner / "runtime.mm", out / "runtime.mm", runtime_variant)
        commands = relocate(json.loads((owner / "commands.json").read_text()), owner, out)
        final = out / "libmetalrobo.dylib"
        parents = [owner / "runtime.mm"]
    else:
        owner = previous / "guarded-dispatch-app-02"
        patch_file(owner / "app-runtime.mm", out / "app-runtime.mm", runtime_variant)
        commands = relocate(json.loads((owner / "commands.json").read_text()), owner, out)
        final = out / "numi-human-native"
        parents = [owner / "app-runtime.mm"]
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with (out / "build.log").open("xb") as log:
        for command in commands:
            subprocess.run(command, cwd=source, stdout=log, stderr=log, check=True, timeout=240)
    (out / "identity.json").write_text(json.dumps({
        "input": {str(p): digest(p) for p in parents},
        "output": {str(final): digest(final)},
    }, indent=2) + "\n")
    for path in sorted(out.glob("*.air")) + sorted(out.glob("*.o")):
        path.unlink()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("shader", "library", "app"))
    for name in ("previous", "source", "out"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    build(**vars(parser.parse_args()))
