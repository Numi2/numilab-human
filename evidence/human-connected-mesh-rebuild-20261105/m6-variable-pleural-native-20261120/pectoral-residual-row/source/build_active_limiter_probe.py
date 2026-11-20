"""Build the existing Matter runtime with a bounded row witness at one microstep."""
import difflib
import json
from pathlib import Path
import subprocess

root = Path.cwd()
previous = root.parent / 'resting-clearance-and-atrium-007/shared-fascia-pressure-build'
commands_path = root.parent / 'resting-clearance-and-atrium-008/pectoral-alpha-trace-runtime/build-commands.json'
build = root / 'active-limiter-trace-runtime'
build.mkdir()
base = (previous / 'runtime.mm').read_text()
trace = r'''
            // Observation of one identified pectoral FEM row. Never changes
            // a solver buffer or affects the success criteria.
            const auto capturePectoralRow = [&](const char* stage) -> bool {
                constexpr NSUInteger node = 196u;
                const bool selected = request.controlStep == 4u && microtick == 61u &&
                    state.dispatch.environmentCount == 1u &&
                    state.dispatch.femNodeCount > node &&
                    std::getenv("NM_FEM_PECTORAL_ROW_TRACE") != nullptr;
                if (!selected) return true;
                using Vector = nm_float4;
                struct Snapshot {
                    Vector residual;
                    Vector solution;
                    NMFGMRESStateGPU solver;
                    NMMatterStatusGPU status;
                    float alpha;
                };
                id<MTLBuffer> snapshot = [state.device newBufferWithLength:sizeof(Snapshot)
                    options:MTLResourceStorageModeShared];
                if (snapshot == nil) return true;
                [encoder endEncoding];
                encoder = nil;
                id<MTLBlitCommandEncoder> copy = [commandBuffer blitCommandEncoder];
                if (copy != nil) {
                    const auto observe = [&](id<MTLBuffer> buffer, NSUInteger src,
                                             NSUInteger dst, NSUInteger size) {
                        [copy copyFromBuffer:buffer sourceOffset:src toBuffer:snapshot
                           destinationOffset:dst size:size];
                    };
                    observe(state.femResidual, node*sizeof(Vector), offsetof(Snapshot,residual), sizeof(Vector));
                    observe(state.femSolution, node*sizeof(Vector), offsetof(Snapshot,solution), sizeof(Vector));
                    observe(state.fgmresStates, 0u, offsetof(Snapshot,solver), sizeof(NMFGMRESStateGPU));
                    observe(state.statuses, 0u, offsetof(Snapshot,status), sizeof(NMMatterStatusGPU));
                    observe(state.environmentLineSearch, 0u, offsetof(Snapshot,alpha), sizeof(float));
                    [copy endEncoding];
                    const auto iteration = micro.solverIteration;
                    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
                        if (finished.status != MTLCommandBufferStatusCompleted) return;
                        const auto& sample = *static_cast<const Snapshot*>(snapshot.contents);
                        const auto& r = sample.residual;
                        const auto& d = sample.solution;
                        const auto& n = sample.solver.nonlinear;
                        const auto& l = sample.solver.diagnostics;
                        std::fprintf(stderr, "pectoral_row stage=%s iter=%u status=%u "
                            "rhs=(%.9g,%.9g,%.9g) correction=(%.9g,%.9g,%.9g) "
                            "alpha=%.9g nonlinear=(%.9g,%.9g,%.9g,%.9g) "
                            "linear=(%.9g,%.9g,%.9g,%.9g)\n",
                            stage,iteration,sample.status.code,r.x,r.y,r.z,d.x,d.y,d.z,
                            sample.alpha,n.x,n.y,n.z,n.w,l.x,l.y,l.z,l.w);
                    }];
                }
                encoder = [commandBuffer computeCommandEncoder];
                if (encoder == nil) {
                    diagnostics.message = "failed to resume Matter after pectoral row capture";
                    ownership->preDynamicsOpen = false;
                    return false;
                }
                return true;
            };
'''
changes = (
    ('            const auto encodeFEMNewtonTrace', trace + '            const auto encodeFEMNewtonTrace'),
    ('            beginSolverStage("matter_newton_linear");',
     '            if (!capturePectoralRow("assembled_rhs")) return diagnostics;\n'
     '            beginSolverStage("matter_newton_linear");'),
    ('            if (!encodeHumanSupportTrace("line_search", false) ||',
     '            if (!capturePectoralRow("selected_step")) return diagnostics;\n'
     '            if (!encodeHumanSupportTrace("line_search", false) ||'),
    ('            if (!encodeFEMNewtonCertificateTrace()) {',
     '            if (!capturePectoralRow("fresh_certificate")) return diagnostics;\n'
     '            if (!encodeFEMNewtonCertificateTrace()) {'),
)
start = base.index('                const bool boundedHumanAlphaTrace =')
stop = base.index(';', start) + 1
updated = base[:start] + '''                const bool boundedHumanAlphaTrace =
                    environments == 1u && state.dispatch.femNodeCount <= 524288u &&
                    objectTotal <= 512u && microtick == 61u;''' + base[stop:]
for anchor, replacement in changes:
    assert updated.count(anchor) == 1, anchor
    updated = updated.replace(anchor, replacement)
unit = build / 'runtime.mm'
unit.write_text(updated)
(root / 'active-limiter-trace.patch').write_text(''.join(difflib.unified_diff(
    base.splitlines(keepends=True), updated.splitlines(keepends=True),
    'a/matter/src/runtime.mm', 'b/matter/src/runtime.mm')))
compiler = json.loads(commands_path.read_text())[0]
object_file = build / 'runtime.mm.o'
library = build / 'libmetalrobo.dylib'
compiler[compiler.index('-c')+1] = str(unit)
compiler[compiler.index('-o')+1] = str(object_file)
link = json.loads((previous.parent / 'shared-fascia-rebase-build/lib/link-command.json').read_text())
command = link['argv']
command[command.index('-o')+1] = str(library)
index, = (i for i, arg in enumerate(command) if arg.endswith('/runtime.mm.o'))
command[index] = str(object_file)
(build / 'build-commands.json').write_text(json.dumps([compiler, command], indent=2) + '\n')
with (root / 'active-limiter-trace-build.log').open('wb') as log:
    for argv in (compiler, command):
        subprocess.run(argv, cwd=link['cwd'], stdout=log, stderr=subprocess.STDOUT, check=True)
object_file.unlink()
print(library)
