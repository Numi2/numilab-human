"""Build the retained native transaction owner with certified staged solver budgets."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import subprocess

root = Path.cwd()
base = root.parent / "resting-clearance-and-atrium-014/certified-budget-complete-audit-build"
build = root / "staged-solver-build"
build.mkdir()
name = "numilab_human_myosim_visual_probe.mm"
before = (base / name).read_text()

start = before.index('                constexpr std::uint32_t shortBudget = 2u;')
end = before.index('\n            }\n#endif\n            else {', start)
assert before[start:end].count("certified Newton budget trial requires shared Matter") == 1
replacement = r'''                constexpr std::uint32_t shortBudget = 2u;
                constexpr std::uint32_t middleBudget = 4u;
                constexpr std::uint32_t shortLinearBudget = 16u;
                const auto normalBudget = runtime.newtonIterationBudget();
                const auto normalLinearBudget = runtime.fgmresIterationBudget();
                require(continuumTransaction->runtime == &runtime && adapter != nullptr &&
                            normalBudget > middleBudget &&
                            normalLinearBudget > shortLinearBudget,
                        "staged Newton trial requires the full shared Matter budgets");
                const auto fingerprint = horizonInput.residentContinuation.previousTransactionFingerprint;
                const auto generation = horizonInput.residentContinuation.previousPhysicsGeneration;
                numi::human::transaction_probe::PhysicalSnapshot physical(
                    coupling.physiology.device, model, horizonInput);
                const auto beforeBody = numi::human::transaction_probe::observePhysicalState(
                    context, physical, fingerprint, generation);
                const auto beforeAdapter = adapter->snapshot();
                const auto beforeMemory = numi::human::transaction_probe::captureCouplingMemory(coupling);
                require(beforeAdapter.available,
                        "failed to inspect the trial's accepted predecessor");

                struct Budget {
                    std::uint32_t newton;
                    std::uint32_t linear;
                };
                const Budget budgets[] = {
                    {shortBudget, shortLinearBudget},
                    {middleBudget, normalLinearBudget},
                    {normalBudget, normalLinearBudget},
                };
                // A contact event that needed the intermediate tier often
                // persists into the next physical step. Restart each new
                // horizon with the shortest budget.
                const std::size_t startTier = lastStagedSolverTier == 0u ? 0u : 1u;
                for (std::size_t tier = startTier; tier < std::size(budgets); ++tier) {
                    const auto budget = budgets[tier];
                    require(runtime.setNewtonIterationBudget(budget.newton) &&
                                runtime.setFGMRESIterationBudget(budget.linear),
                            "the current accepted predecessor did not release Matter for dispatch");
                    const auto beforeMatter = runtime.snapshot();
                    require(beforeMatter.available, "failed to inspect accepted Matter state");
                    segmentResult = {};
                    segmentDiagnostics = context.run(model, horizonInput, segmentResult);
                    std::cout << "human_staged_solver_attempt step=" << completedSteps + 1u
                              << " tier=" << tier << " newton=" << budget.newton
                              << " linear=" << budget.linear
                              << " accepted=" << segmentDiagnostics.succeeded()
                              << " published=" << segmentDiagnostics.published
                              << " elapsed_ms=" << segmentDiagnostics.elapsedMilliseconds << std::endl;
                    if (segmentDiagnostics.succeeded() && segmentDiagnostics.published) {
                        lastStagedSolverTier = tier;
                        require(runtime.setNewtonIterationBudget(normalBudget) &&
                                    runtime.setFGMRESIterationBudget(normalLinearBudget),
                                "unable to restore solver budgets after acceptance");
                        break;
                    }

                    if (tier + 1u == std::size(budgets) ||
                        !segmentDiagnostics.dispatched || segmentDiagnostics.published ||
                        segmentDiagnostics.firstStandGPUStatusCode != MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED ||
                        segmentDiagnostics.completedStandSteps != completedSteps ||
                        segmentDiagnostics.residentStateGeneration != 0u ||
                        segmentDiagnostics.residentStateTransactionFingerprint != 0u) {
                        break;
                    }
                    const auto afterMatter = runtime.snapshot();
                    if (!afterMatter.available || afterMatter.statuses.size() != 1u ||
                        afterMatter.statuses.front().code != NM_STATUS_NONLINEAR_SOLVER_FAILURE)
                        break;

                    const auto afterBody = numi::human::transaction_probe::observePhysicalState(
                        context, physical, fingerprint, generation);
                    const auto afterAdapter = adapter->snapshot();
                    const auto afterMemory = numi::human::transaction_probe::captureCouplingMemory(coupling);
                    require(afterAdapter.available &&
                                numi::human::transaction_probe::samePhysicalState(beforeBody, afterBody) &&
                                numi::human::transaction_probe::sameMatterAcceptedState(beforeMatter, afterMatter) &&
                                numi::human::transaction_probe::sameMatterCirculation(beforeMatter, afterMatter) &&
                                numi::human::transaction_probe::sameTendonAdapterAcceptedHistory(beforeAdapter, afterAdapter) &&
                                numi::human::transaction_probe::sameAcceptedCouplingMemory(beforeMemory, afterMemory),
                            "staged solver retry found a changed accepted state");
                    rejectedAttemptMilliseconds += segmentDiagnostics.elapsedMilliseconds;
                    newtonRejectedSteps.push_back(completedSteps);
                    std::cout << "human_staged_solver_rollback step=" << completedSteps + 1u
                              << " tier=" << tier << " rollback=byte_identical"
                              << " residual=" << afterMatter.statuses.front().diagnostics.x << std::endl;
                }'''
source = before[:start] + replacement + before[end:]
scope = '        const std::uint32_t requestedSteps = horizonInput.stand.stepCount;'
assert source.count(scope) == 1
source = source.replace(scope, scope+'\n        [[maybe_unused]] std::size_t lastStagedSolverTier = 0u;')
anchor = '                if (newtonRejectedSteps != nullptr &&\n'
assert source.count(anchor) == 1
source = source.replace(anchor, '                while (newtonRejectedSteps != nullptr &&\n')
(build/name).write_text(source)
(root/"staged-solver-budgets.patch").write_text("".join(difflib.unified_diff(
    before.splitlines(keepends=True), source.splitlines(keepends=True),
    "a/apps/"+name, "b/apps/"+name)))
for p in (*base.glob("*.hpp"), base/"runtime.mm.o"):
    os.link(p, build/p.name)
(build/"metalrobo").symlink_to(base/"metalrobo", target_is_directory=True)
commands = [[part.replace(str(base)+"/", str(build)+"/") for part in cmd]
            for cmd in json.loads((base/"build-commands.json").read_text())]
(build/"build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
with (root/"staged-solver-build.log").open("wb") as log:
    for cmd in commands:
        subprocess.run(cmd, cwd=Path.home()/"Work/skin-fat-native-integration-209/build-native",
                       stdout=log, stderr=subprocess.STDOUT, check=True)
with (build/"numi-human-native").open("rb") as stream:
    print(hashlib.file_digest(stream,"sha256").hexdigest())
