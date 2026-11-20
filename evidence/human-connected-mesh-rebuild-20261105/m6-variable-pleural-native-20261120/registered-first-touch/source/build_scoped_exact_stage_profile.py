"""Build an isolated Metal host with bounded stage counter collection."""
import difflib
import json
from pathlib import Path
import subprocess

root=Path.cwd()
n=root.parent/'resting-clearance-and-atrium-008'
build=root/'scoped-exact-stage-runtime'
build.mkdir(exist_ok=True)
stage_source=n.parent/'resting-clearance-and-atrium-007/shared-fascia-pressure-build/runtime.mm'
base=stage_source.read_text()

first=base.index('        const bool measureSolverStages = [] {')
last=base.index('        detail::HostStageTiming hostStageTiming',first)
replacement='''        // The timer's root selector must also gate encoder splitting. Full
        // roots at high FEM rates can exhaust Metal's counter sample buffer;
        // the optional microtick selector captures a complete bounded slice.
        const auto timingIndex = [](const char* name, unsigned fallback) {
            const char* setting = std::getenv(name);
            if (setting == nullptr) return fallback;
            if (*setting < '0' || *setting > '9') return ~0u;
            char* end = nullptr;
            const unsigned long value = std::strtoul(setting, &end, 10);
            return *end == '\\0' && value < ~0u
                ? static_cast<unsigned>(value) : ~0u;
        };
        const char* stageSetting = std::getenv("NUMI_MATTER_GPU_TIMING_STAGES");
        const char* counterSetting = std::getenv("NUMI_MATTER_GPU_TIMING");
        const unsigned timingRoot = timingIndex("NUMI_MATTER_GPU_TIMING_ROOT", 100u);
        const char* tickSetting = std::getenv("NUMI_MATTER_GPU_TIMING_MICROTICK");
        const unsigned timingMicrotick = timingIndex("NUMI_MATTER_GPU_TIMING_MICROTICK", ~0u);
        const bool measureSolverStages =
            stageSetting != nullptr && std::strcmp(stageSetting, "1") == 0 &&
            counterSetting != nullptr && std::strcmp(counterSetting, "1") == 0 &&
            timingRoot != ~0u && timingRoot == request.controlStep &&
            (tickSetting == nullptr || timingMicrotick != ~0u);
        bool measureSolverMicrotick = false;
        if (measureSolverStages) {
            std::fprintf(stderr,
                "matter_gpu_stage_timing_requested root=%u selector_microtick=%s physics_substep=%u\\n",
                request.controlStep, tickSetting != nullptr ? tickSetting : "all",
                request.physicsSubstep);
        }
'''
text=base[:first]+replacement+base[last:]
text=text.replace('            if (!measureSolverStages && !gpuProgress.enabled()) return;',
                  '            if (!measureSolverMicrotick && !gpuProgress.enabled()) return;',1)
text=text.replace('            encoder = measureSolverStages\n                ? detail::timingSpanBoundary',
                  '            encoder = measureSolverMicrotick\n                ? detail::timingSpanBoundary',1)
needle='''             ++microtick) {
            NMMicrostepGPU micro{};'''
assert text.count(needle)==1
text=text.replace(needle,'''             ++microtick) {
            measureSolverMicrotick = measureSolverStages &&
                (tickSetting == nullptr || microtick == timingMicrotick);
            NMMicrostepGPU micro{};''',1)
unit=build/'runtime.mm'
unit.write_text(text)
patch=''.join(difflib.unified_diff(base.splitlines(keepends=True),text.splitlines(keepends=True),
    'a/matter/src/runtime.mm','b/matter/src/runtime.mm'))
(root/'scoped-exact-stage-profile.patch').write_text(patch)
commands=json.loads((n/'pectoral-alpha-trace-runtime/build-commands.json').read_text())
obj=build/'runtime.mm.o'
lib=build/'libmetalrobo.dylib'
compiler=commands[0]
compiler[compiler.index('-o')+1]=str(obj)
compiler[compiler.index('-c')+1]=str(unit)
p=root.parent/'resting-clearance-and-atrium-007/shared-fascia-rebase-build/lib/link-command.json'
link=json.loads(p.read_text())
cmd=link['argv']
cmd[cmd.index('-o')+1]=str(lib)
swap=[i for i,arg in enumerate(cmd) if arg.endswith('/runtime.mm.o')]
assert len(swap)==1
cmd[swap[0]]=str(obj)
(build/'build-commands.json').write_text(json.dumps([compiler,cmd],indent=2)+'\n')
for call in (compiler,cmd):
    subprocess.run(call,cwd=link['cwd'],check=True)
print(lib,flush=True)
