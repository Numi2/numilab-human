"""Extend the pinned runtime observer to every object at one exact microtick."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent / "resting-clearance-and-atrium-013/glide-refresh-build"
OUTPUT = HERE / "backtracking-witness-build"

original = OWNER / "runtime.mm"
base = original.read_text()
start = base.index("            const char* backtrackingObjectSetting")
end = base.index('            if (!encodeVascularTrace("after_fem_backtracking"))', start)
observer = base[start:end]
observer = observer.replace(
    '            const std::uint32_t backtrackingNewton = backtrackingNewtonSetting == nullptr',
    '            const char* backtrackingTickSetting = std::getenv("NM_FEM_BACKTRACKING_TRACE_MICROTICK");\n'
    '            const bool selectedBacktrackingTick = backtrackingTickSetting == nullptr ||\n'
    '                microtick == std::strtoul(backtrackingTickSetting, nullptr, 10);\n'
    '            const bool allBacktrackingNewton = backtrackingNewtonSetting != nullptr &&\n'
    '                std::strcmp(backtrackingNewtonSetting, "all") == 0;\n'
    '            const std::uint32_t backtrackingNewton = backtrackingNewtonSetting == nullptr')
observer = observer.replace(
    '                micro.solverIteration == backtrackingNewton && environments == 1u &&',
    '                selectedBacktrackingTick && environments == 1u &&\n'
    '                (allBacktrackingNewton || micro.solverIteration == backtrackingNewton) &&')
observer = observer.replace(
    '                const std::uint32_t objectIndex = std::strtoul(backtrackingObjectSetting,nullptr,10);\n'
    '                if (objectIndex >= state.objectLayout.size()) {',
    '                const bool allObjects = std::strcmp(backtrackingObjectSetting, "all") == 0;\n'
    '                const auto firstObject = allObjects ? 0u :\n'
    '                    std::uint32_t(std::strtoul(backtrackingObjectSetting,nullptr,10));\n'
    '                if (firstObject >= state.objectLayout.size()) {')
observer_start = observer.index("                const auto object = state.objectLayout[objectIndex];")
observer_end = observer.rindex("            }")
observer = (
    observer[:observer_start] +
    '                const auto lastObject = allObjects ? state.objectLayout.size() : firstObject + 1u;\n'
    '                for (auto objectIndex = firstObject; objectIndex < lastObject; ++objectIndex) {\n' +
    "".join("    " + line if line.strip() else line
            for line in observer[observer_start:observer_end].splitlines(keepends=True)) +
    '                }\n' + observer[observer_end:])
observer = observer.replace(
    "                        const auto root=request.controlStep;",
    "                        const auto root=request.controlStep;\n"
    "                        const auto tick=microtick;")
observer = observer.replace('<<root<<",\\"iteration\\":"<<iteration',
                            '<<root<<",\\"microtick\\":"<<tick<<",\\"iteration\\":"<<iteration')
assert "selectedBacktrackingTick" in observer and "<<tick<<" in observer
assert "for (auto objectIndex" in observer
updated = base[:start] + observer + base[end:]

OUTPUT.mkdir(exist_ok=True)
(OUTPUT / "runtime.mm").write_text(updated)
(OUTPUT / "backtracking-witness.patch").write_text("".join(difflib.unified_diff(
    base.splitlines(keepends=True), updated.splitlines(keepends=True),
    "a/matter/src/runtime.mm", "b/matter/src/runtime.mm")))
recipe = json.loads((OWNER / "build-commands.json").read_text())
base_human_source = str(OWNER / "numi_human.mm")
recipe = [[word if word == base_human_source else word.replace(str(OWNER) + "/", str(OUTPUT) + "/")
           for word in command] for command in recipe]
cwd = json.loads((HERE.parent / "resting-clearance-and-atrium-007/shared-fascia-rebase-build/lib/link-command.json")
                 .read_text())["cwd"]
(OUTPUT / "build.json").write_text(json.dumps({"argvs": recipe, "cwd": cwd,
    "input_sha256": hashlib.sha256(original.read_bytes()).hexdigest()}, indent=2) + "\n")
with (OUTPUT / "build.log").open("wb") as log:
    for command in recipe:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
for name in ("runtime.mm.o", "numi_human.mm.o"):
    (OUTPUT / name).unlink()
