"""Build a bounded GPU-stage trace for the mismatched line-search receipt."""
import difflib
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
previous = here.with_name("resting-clearance-and-atrium-007")
scratch = Path("/Volumes/NumiHumanTrial")
out = here / "pectoral-limiter-trace-runtime"
out.mkdir()
pressure = previous / "shared-fascia-pressure-build"
rebase = previous / "shared-fascia-rebase-build"
old_path = pressure / "runtime.mm"
new_path = out / "runtime.mm"
original = old_path.read_text()
before = '''                const char* traceRoot = std::getenv("NM_VASCULAR_TRACE_ROOT");
                if (traceRoot == nullptr || vascularTotal == 0u || vascularTotal > 4096u ||
                    request.controlStep != std::strtoul(traceRoot, nullptr, 10)) return true;'''
after = before + '''
                if (microtick != 12u) return true;
                const std::string_view name(stage);
                if (name != "after_fem_backtracking" &&
                    name != "after_deformable_limit" &&
                    name != "before_support_limit" &&
                    name != "after_validating_line_search" &&
                    name != "certificate") return true;'''
assert original.count(before)==1
text = original.replace(before, after)
marker = '''            dispatchGroups32(
                "nm_human_support_validate_final_line_search",'''
assert text.count(marker)==1
text = text.replace(marker, '''            if (!encodeVascularTrace("after_vascular_line_search")) {
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }
''' + marker)
marker = '''            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (femCapsuleResidualArmijoEnabled ||
                femCapsuleBoxResidualArmijoEnabled ||
                femPureMultiFEMFullContactResidualArmijoEnabled) {
                id<MTLBuffer> armijoContactSamples ='''
assert text.count(marker)==1
text = text.replace(marker, '''            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (!encodeVascularTrace("after_validating_line_search")) {
                ownership->preDynamicsOpen = false;
                return diagnostics;
            }
''' + marker.split("\n",1)[1])
old_terminal = """                if ((!boundedTerminalOnly&&!selectedFEMNewtonTraceRoot(root)) || microtick != 0u)
                    return true;"""
new_terminal = """                if (!boundedTerminalOnly || request.controlStep != 4u || microtick != 12u)
                    return true;"""
assert text.count(old_terminal) == 1
text = text.replace(old_terminal, new_terminal)
new_path.write_text(text)
(here / "pectoral-limiter-trace.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True), text.splitlines(keepends=True),
    "a/matter/src/runtime.mm", "b/matter/src/runtime.mm")))
obj = scratch / "pectoral-limiter-trace.o"
compile = json.loads((pressure / "build-commands.json").read_text())[0]
compile[compile.index("-o") + 1] = str(obj)
compile[compile.index("-c") + 1] = str(new_path)
lib = out / "libmetalrobo.dylib"
link_record = json.loads((rebase / "lib/link-command.json").read_text())
link = link_record["argv"]
link[link.index(str(rebase / "runtime.mm.o"))] = str(obj)
link[link.index("-o")+1] = str(lib)
(out / "build-commands.json").write_text(json.dumps([compile,link],indent=2)+"\n")
subprocess.run(compile, cwd=link_record["cwd"], check=True)
subprocess.run(link, cwd=link_record["cwd"], check=True)
obj.unlink()
