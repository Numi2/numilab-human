"""Build a gated FGMRES row transform on the exact retained native sources."""
import difflib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "fgmres-patch-linearization-019"
OLD = PARENT / "build"
BUILD = HERE / "build"

def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one source anchor for {before[:100]}")
    return text.replace(before, after, 1)

def write_change(name, source, updated, label):
    target = BUILD / name
    target.write_text(updated)
    patch = difflib.unified_diff(
        source.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="a/" + label, tofile="b/" + label, n=0)
    (HERE / (target.name + ".diff")).write_text("".join(patch))

def runtime(s):
    s = replace(s, """        const auto applyLeftRowEquilibration = [&](
            id<MTLBuffer> source, id<MTLBuffer> destination) {""", """        const bool boundRowTransformWork = [] {
            const char* value = std::getenv("NM_FGMRES_ROW_TRANSFORM_WORK_DOMAIN");
            return value != nullptr && std::strcmp(value, "1") == 0;
        }();
        const auto applyLeftRowEquilibration = [&](
            id<MTLBuffer> source, id<MTLBuffer> destination,
            bool arnoldiImage = false) {""")
    start = s.index("        const auto applyLeftRowEquilibration =")
    end = s.index("\n        };", start)
    snippet = s[start:end]
    snippet = replace(snippet, """            dispatchThreads(
                "nm_fgmres_apply_left_row_equilibration",
                state.fgmresLayout.unknownCount, [&] {""", """            const bool nodeOnly =
                boundRowTransformWork && arnoldiImage && source == destination;
            const NSUInteger rowCount = nodeOnly
                ? environments * state.dispatch.femNodeCount
                : state.fgmresLayout.unknownCount;
            dispatchThreads("nm_fgmres_apply_left_row_equilibration", rowCount, [&] {""")
    snippet = replace(snippet, """                [encoder setBuffer:destination offset:0u atIndex:3u];
            });""", """                [encoder setBuffer:destination offset:0u atIndex:3u];
            }, nodeOnly ? WorkDomain::femNodes : WorkDomain::generic);""")
    s = s[:start] + snippet + s[end:]
    s = replace(s, """                    dispatchThreads("nm_fgmres_unscale_preconditioner_input",
                        state.fgmresLayout.unknownCount, [&] {""", """                    const std::uint32_t useNodeDomain =
                        boundRowTransformWork && fgmresColumnDispatch &&
                        state.dispatch.femNodeCount >= 256u ? 1u : 0u;
                    const NSUInteger rowCount = useNodeDomain
                        ? state.dispatch.femNodeCount
                        : state.fgmresLayout.unknownCount;
                    dispatchThreads("nm_fgmres_unscale_preconditioner_input",
                        rowCount, [&] {""")
    s = replace(s, """                        [encoder setBuffer:state.femLeftRowEquilibrationResidual
                                     offset:0u atIndex:3u];
                    });
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    preconditionerInputBasis""", """                        [encoder setBuffer:state.femLeftRowEquilibrationResidual
                                     offset:0u atIndex:3u];
                        [encoder setBytes:&useNodeDomain
                                     length:sizeof(useNodeDomain) atIndex:4u];
                    }, useNodeDomain ? WorkDomain::femNodes : WorkDomain::generic);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    preconditionerInputBasis""")
    return replace(s, """                applyLeftRowEquilibration(state.femOperatorValue,
                                           state.femOperatorValue);""", """                applyLeftRowEquilibration(state.femOperatorValue,
                                           state.femOperatorValue, true);""")

def shader(s):
    start = s.index("kernel void nm_fgmres_unscale_preconditioner_input(")
    end = s.index("\nkernel void", start + 1)
    snippet = s[start:end]
    snippet = replace(snippet,
        """    device float4* destination [[buffer(3)]],
    uint global""",
        """    device float4* destination [[buffer(3)]],
    constant uint& useNodeDomain [[buffer(4)]],
    uint global""")
    snippet = replace(snippet,
        """    if (global >= layout.unknownCount) return;
    float4 value = source[global];
    const uint totalNodes = dispatch.environmentCount * dispatch.femNodeCount;
    if (global < totalNodes) value.xyz /= rowScale[global];
    destination[global] = value;""",
        """    const uint totalNodes = dispatch.environmentCount * dispatch.femNodeCount;
    if (global >= (useNodeDomain != 0u ? totalNodes : layout.unknownCount)) return;
    float4 value = source[global];
    if (global < totalNodes) value.xyz /= rowScale[global];
    destination[global] = value;
    if (useNodeDomain != 0u) {
        // The vector still has its complete original layout. Copy every
        // identity-scaled row from the same Arnoldi column, including tails.
        for (ulong row = ulong(global) + totalNodes; row < layout.unknownCount;
             row += totalNodes)
            destination[row] = source[row];
    }""")
    return s[:start] + snippet + s[end:]

def main():
    BUILD.mkdir(exist_ok=True)
    shutil.copytree(OLD / "src", BUILD / "src", dirs_exist_ok=True)
    for name in ["runtime.mm", "app-runtime.mm"]:
        text = (OLD / name).read_text()
        write_change(name, text, runtime(text), "matter/src/runtime.mm")
    name = "src/metal/fgmres.metalinc"
    text = (OLD / name).read_text()
    write_change(name, text, shader(text), "matter/src/metal/fgmres.metalinc")
    commands = json.loads((PARENT / "build-commands.json").read_text())
    commands = [[arg.replace(str(OLD), str(BUILD)) for arg in cmd] for cmd in commands]
    (HERE / "row-transform-build-commands.json").write_text(
        json.dumps(commands, indent=2) + "\n")
    for cmd in commands:
        print(cmd[0], cmd[-1], flush=True)
        subprocess.run(cmd, cwd=Path.home() / "Work/skin-fat-native-integration-209/build-native",
                       check=True)

if __name__ == "__main__":
    main()
