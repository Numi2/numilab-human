from pathlib import Path
import difflib
import json
import subprocess

root = Path.cwd()
base = root/"glide-refresh-build"
build = root/"glide-refresh-witness-build"
build.mkdir(exist_ok=True)
text = (base/"runtime.mm").read_text()
observer = (root/"pectoral-energy-observer.inc").read_text()
observer = observer.replace("node = 194u, nodeCount = 38u", "node = 270u, nodeCount = 88u")
observer = observer.replace("element = 270u, elementCount = 54u", "element = 378u, elementCount = 129u")
observer = observer.replace("request.controlStep > 4u", "request.controlStep != 7u")
observer = observer.replace("request.controlStep == 4u && microtick >= 58u && microtick <= 61u",
                            "microtick >= 81u && microtick <= 85u")
changes = [
    ('            const auto encodeFEMNewtonTrace',
     observer+'            const auto encodeFEMNewtonTrace'),
    ('            beginSolverStage("matter_newton_linear");',
     '            if (!capturePectoralEnergy("assembled_rhs")) return diagnostics;\n'
     '            beginSolverStage("matter_newton_linear");'),
    ('            if (!encodeFEMNewtonCertificateTrace()) {',
     '            if (!capturePectoralEnergy("fresh_certificate")) return diagnostics;\n'
     '            if (!encodeFEMNewtonCertificateTrace()) {'),
    ('                    state.femFullCoupledTrueResidualEnabled &&\n'
     '                    environments == 1u && state.dispatch.femNodeCount <= 524288u &&\n'
     '                    objectTotal <= 512u && microtick <= 4u;',
     '                    environments == 1u && objectTotal == 14u && microtick == 85u;'),
    ('                micro.solverIteration == backtrackingNewton && environments == 1u &&',
     '                microtick == 85u && micro.solverIteration == backtrackingNewton && environments == 1u &&'),
]
updated = text
for source,target in changes:
    assert updated.count(source) == 1, source
    updated = updated.replace(source,target)
(build/"runtime.mm").write_text(updated)
(root/"glide-refresh-failure-observer.patch").write_text("".join(difflib.unified_diff(
    text.splitlines(keepends=True),updated.splitlines(keepends=True),
    "a/matter/src/runtime.mm","b/matter/src/runtime.mm")))
commands = [[x.replace(str(base)+"/",str(build)+"/") if "numi_human.mm" not in x else x
             for x in cmd] for cmd in json.loads((base/"build-commands.json").read_text())]
link_info = json.loads((root.parent/"resting-clearance-and-atrium-007"
    /"shared-fascia-rebase-build/lib/link-command.json").read_text())
with (root/"glide-refresh-failure-build.log").open("wb") as log:
    for cmd in commands:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=link_info["cwd"],check=True)
for p in (build/"runtime.mm.o",base/"numi_human.mm.o"):
    p.unlink()

runner = (root/"run_pectoral_load_probe.py").read_text()
runner = runner.replace("root / 'pectoral-load-trial'", "root / 'glide-refresh-witness-trial'")
runner = runner.replace("root.parent / 'resting-clearance-and-atrium-012' / 'residual-bank-shader/NumiMatter.metallib'",
                        "root / 'glide-refresh-build/shader/NumiMatter.metallib'")
runner = runner.replace("root / 'pectoral-load-trace-runtime/libmetalrobo.dylib'",
                        "root / 'glide-refresh-witness-build/libmetalrobo.dylib'")
runner = runner.replace("NM_FEM_PECTORAL_ROW_TRACE='1'", """NM_HUMAN_GLIDE_REFRESH='1',
    NM_FEM_PECTORAL_ENERGY_TRACE=str(run / "energy.csv"),
    NM_FEM_NEWTON_TRACE_ROOT='7',
    NM_FEM_BACKTRACKING_TRACE_ROOT='7',
    NM_FEM_BACKTRACKING_TRACE_NEWTON='3',
    NM_FEM_BACKTRACKING_TRACE_OBJECT='7'""")
(root/"run_refreshed_failure_probe.py").write_text(runner)
