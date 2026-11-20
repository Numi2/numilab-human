"""Build an isolated owner-level retry trial on the retained first-touch native scene."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import subprocess

root = Path.cwd()
base = root.parent/"resting-clearance-and-atrium-011/registered-first-touch-build"
build = root/"certified-budget-build"
build.mkdir()
name = "numilab_human_myosim_visual_probe.mm"
original = (base/name).read_text()
anchor = """            } else {
                segmentDiagnostics = context.run(
                    model, horizonInput, segmentResult);
            }
            const auto physicalEnd = std::chrono::steady_clock::now();"""
patched = r"""            }
#ifdef NUMI_HUMAN_RESTING_SCENE
            else if (completedSteps != 0u && restingProgram != nullptr &&
                     std::getenv("NUMI_HUMAN_CERTIFIED_NEWTON_BUDGET_TRIAL") != nullptr) {
                // The retry lives at the authoritative transaction boundary.
                // The selected budget changes iteration work only. All device
                // certificates, forces, and exactly synchronized states remain owned
                // by the unchanged native transaction.
                require(segmentSteps == 1u && horizonInput.residentContinuation.valid() &&
                            restingCouplingContext != nullptr && continuumTransaction != nullptr,
                        "certified Newton budget trial requires a single accepted predecessor");
                auto& coupling = *static_cast<NumiHumanRestingCoupling*>(restingCouplingContext);
                auto& runtime = coupling.physiology.runtime;
                auto* adapter = continuumTransaction->adapter;
                constexpr std::uint32_t shortBudget = 2u;
                const std::uint32_t normalBudget = runtime.newtonIterationBudget();
                require(continuumTransaction->runtime == &runtime && adapter != nullptr &&
                            normalBudget > shortBudget,
                        "certified Newton budget trial requires shared Matter and a larger fallback budget");
                const auto fingerprint = horizonInput.residentContinuation.previousTransactionFingerprint;
                const auto generation = horizonInput.residentContinuation.previousPhysicsGeneration;
                numi::human::transaction_probe::PhysicalSnapshot physical(
                    coupling.physiology.device, model, horizonInput);
                require(runtime.setNewtonIterationBudget(shortBudget),
                        "failed to set certified trial budget before dispatch");
                const auto beforeBody = numi::human::transaction_probe::observePhysicalState(
                    context, physical, fingerprint, generation);
                const auto beforeMatter = runtime.snapshot();
                const auto beforeAdapter = adapter->snapshot();
                const auto beforeMemory = numi::human::transaction_probe::captureCouplingMemory(coupling);
                require(beforeMatter.available && beforeAdapter.available,
                        "failed to inspect the trial's accepted predecessor");
                segmentDiagnostics = context.run(model, horizonInput, segmentResult);

                if (segmentDiagnostics.dispatched && !segmentDiagnostics.succeeded() &&
                    !segmentDiagnostics.published &&
                    segmentDiagnostics.firstStandGPUStatusCode == MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED &&
                    segmentDiagnostics.completedStandSteps == completedSteps &&
                    segmentDiagnostics.residentStateGeneration == 0u &&
                    segmentDiagnostics.residentStateTransactionFingerprint == 0u) {
                    const auto afterMatter = runtime.snapshot();
                    if (afterMatter.available && afterMatter.statuses.size() == 1u &&
                        afterMatter.statuses.front().code == NM_STATUS_NONLINEAR_SOLVER_FAILURE) {
                        const auto afterBody = numi::human::transaction_probe::observePhysicalState(
                            context, physical, fingerprint, generation);
                        const auto afterAdapter = adapter->snapshot();
                        const auto afterMemory =
                            numi::human::transaction_probe::captureCouplingMemory(coupling);
                        require(afterAdapter.available &&
                                    numi::human::transaction_probe::samePhysicalState(beforeBody, afterBody) &&
                                    numi::human::transaction_probe::sameMatterAcceptedState(beforeMatter, afterMatter) &&
                                    numi::human::transaction_probe::sameMatterCirculation(beforeMatter, afterMatter) &&
                                    numi::human::transaction_probe::sameTendonAdapterAcceptedHistory(beforeAdapter, afterAdapter) &&
                                    numi::human::transaction_probe::sameAcceptedCouplingMemory(beforeMemory, afterMemory),
                                "certified Newton retry found a changed accepted state");
                        require(runtime.setNewtonIterationBudget(normalBudget),
                                "the verified rollback did not release Matter for retry");
                        rejectedAttemptMilliseconds += segmentDiagnostics.elapsedMilliseconds;
                        std::cout << "human_verified_newton_retry step=" << completedSteps + 1u
                                  << " short_budget=" << shortBudget << " retry_budget=" << normalBudget
                                  << " residual=" << afterMatter.statuses.front().diagnostics.x
                                  << " rollback=byte_identical" << std::endl;
                        segmentResult = {};
                        segmentDiagnostics = context.run(model, horizonInput, segmentResult);
                        std::cout << "human_verified_newton_retry_outcome step=" << completedSteps + 1u
                                  << " completed_steps=" << segmentDiagnostics.completedStandSteps
                                  << " accepted=" << segmentDiagnostics.succeeded()
                                  << " published=" << segmentDiagnostics.published << std::endl;
                    }
                }
                if (runtime.newtonIterationBudget() != normalBudget && segmentDiagnostics.succeeded()) {
                    require(runtime.setNewtonIterationBudget(normalBudget),
                            "unable to restore the configured Newton work budget after acceptance");
                }
            }
#endif
            else {
                segmentDiagnostics = context.run(
                    model, horizonInput, segmentResult);
            }
            const auto physicalEnd = std::chrono::steady_clock::now();"""
assert original.count(anchor) == 1
updated = original.replace(anchor, patched)
(build/name).write_text(updated)
(root/"certified-budget-trial.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True), updated.splitlines(keepends=True),
    "a/apps/"+name,"b/apps/"+name)))
for path in (*base.glob("*.hpp"),base/"runtime.mm.o"):
    os.link(path,build/path.name)
os.symlink(base/"metalrobo",build/"metalrobo",target_is_directory=True)
commands = [[part.replace(str(base)+"/",str(build)+"/") for part in cmd]
            for cmd in json.loads((base/"build-commands.json").read_text())]
(build/"build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
with (root/"certified-budget-build.log").open("wb") as out:
    for cmd in commands:
        subprocess.run(cmd,cwd=Path.home()/"Work/skin-fat-native-integration-209/build-native",
                       stdout=out,stderr=subprocess.STDOUT,check=True)
with (build/"numi-human-native").open("rb") as stream:
    print(hashlib.file_digest(stream,"sha256").hexdigest())
