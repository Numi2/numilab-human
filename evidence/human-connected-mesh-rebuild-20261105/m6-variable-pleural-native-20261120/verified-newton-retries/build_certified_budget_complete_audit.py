"""Build the isolated Newton-budget trial with retry-aware tendon audit receipts."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import subprocess

root = Path.cwd()
base = root / "certified-budget-build"
build = root / "certified-budget-complete-audit-build"
build.mkdir()
name = "numilab_human_myosim_visual_probe.mm"
before = (base / name).read_text()
source = before

def replace_once(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new, 1)

replace_once(
    '    std::uint32_t lastAuthoritativeSubmissionSteps = 0u;',
    '''    std::uint32_t lastAuthoritativeSubmissionSteps = 0u;
    std::vector<std::uint32_t> initialNewtonRejectedSteps;
    std::vector<std::uint32_t> replayNewtonRejectedSteps;''',
)
replace_once(
    '''        std::vector<HumanTendonContinuumTransaction::AcceptedStep>*
            capturedSteps
    ) {''',
    '''        std::vector<HumanTendonContinuumTransaction::AcceptedStep>*
            capturedSteps,
        [[maybe_unused]] std::vector<std::uint32_t>& newtonRejectedSteps
    ) {''',
)
replace_once(
    '                        rejectedAttemptMilliseconds += segmentDiagnostics.elapsedMilliseconds;',
    '''                        rejectedAttemptMilliseconds += segmentDiagnostics.elapsedMilliseconds;
                        newtonRejectedSteps.push_back(completedSteps);''',
)
replace_once(
    '            ? nullptr : &continuumTransaction->acceptedSteps);',
    '            ? nullptr : &continuumTransaction->acceptedSteps,\n        initialNewtonRejectedSteps);',
)
replace_once(
    '                ? nullptr : &continuumTransaction->replayedSteps);',
    '                ? nullptr : &continuumTransaction->replayedSteps,\n            replayNewtonRejectedSteps);',
)
replace_once(
    '            stepCount + expectedRejectedAuditPasses;',
    '''            stepCount + expectedRejectedAuditPasses +
            static_cast<std::uint32_t>(initialNewtonRejectedSteps.size());''',
)
replace_once(
    '            expectedAcceptedConsumerPasses + expectedRejectedAuditPasses;',
    '''            expectedAcceptedConsumerPasses + expectedRejectedAuditPasses +
            static_cast<std::uint32_t>(initialNewtonRejectedSteps.size() +
                                       replayNewtonRejectedSteps.size());''',
)
replace_once(
    '''        for (std::uint32_t phase = 0u; phase < consumerPhases; ++phase) {
            for (std::uint32_t step = 0u; step < stepCount; ++step) {''',
    '''        for (std::uint32_t phase = 0u; phase < consumerPhases; ++phase) {
            const auto* newtonRejectedSteps = phase == 0u ? &initialNewtonRejectedSteps :
                (phase == phaseCount ? &replayNewtonRejectedSteps : nullptr);
            std::size_t nextNewtonRejection = 0u;
            for (std::uint32_t step = 0u; step < stepCount; ++step) {
                if (newtonRejectedSteps != nullptr &&
                    nextNewtonRejection < newtonRejectedSteps->size() &&
                    (*newtonRejectedSteps)[nextNewtonRejection] == step) {
                    expectedConsumerStepIndices.push_back(step);
                    ++nextNewtonRejection;
                }''',
)
replace_once(
    '            stepCount + terminalRejectedAuditAttempts;',
    """            stepCount + terminalRejectedAuditAttempts +
            static_cast<std::uint32_t>(initialNewtonRejectedSteps.size());""",
)
replace_once(
    '                std::to_string(terminalRejectedAuditAttempts));',
    """                std::to_string(terminalRejectedAuditAttempts) +
                " verified_newton_retries=" + std::to_string(initialNewtonRejectedSteps.size()));""",
)
(build / name).write_text(source)
(root / "certified-newton-attempt-audits.patch").write_text("".join(difflib.unified_diff(
    before.splitlines(keepends=True), source.splitlines(keepends=True),
    "a/apps/" + name, "b/apps/" + name)))
for f in (*base.glob("*.hpp"), base / "runtime.mm.o"):
    os.link(f, build / f.name)
(build / "metalrobo").symlink_to(base / "metalrobo", target_is_directory=True)
commands = [[part.replace(str(base) + "/", str(build) + "/") for part in cmd]
            for cmd in json.loads((base / "build-commands.json").read_text())]
(build / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
with (root / "certified-budget-complete-audit-build.log").open("wb") as out:
    for cmd in commands:
        subprocess.run(cmd, cwd=Path.home() / "Work/skin-fat-native-integration-209/build-native",
                       stdout=out, stderr=subprocess.STDOUT, check=True)
with (build / "numi-human-native").open("rb") as stream:
    print(hashlib.file_digest(stream, "sha256").hexdigest())
