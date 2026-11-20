"""Build a square-root row-weight candidate without changing the physical residual."""
from pathlib import Path
import json
import shutil
import sys
import subprocess

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "fgmres-endpoint-window-020"
sys.path.insert(0, str(PREVIOUS))
from prepare_row_transform import replace, write_change

# Reuse the native build's diff wrapper, pointing its outputs at this case.
import prepare_row_transform as parent
parent.HERE = HERE
parent.BUILD = HERE / "build"
SOURCE = PREVIOUS / "build"
BUILD = HERE / "build"

def rewrite_runtime(text):
    text = replace(text, "    bool femCoupledRowEquilibrationEnabled = false;",
        """    bool femCoupledRowEquilibrationEnabled = false;
    bool femTemperedRowEquilibrationEnabled = false;""")
    insert_before = "        if (candidate->fgmresFreshRhsPreconditionerPreflightRequested) {"
    text = replace(text, insert_before, """        if (const char* requested = std::getenv("NM_FGMRES_ROW_SCALE_SQRT")) {
            const bool enable = std::strcmp(requested, "1") == 0;
            if ((!enable && std::strcmp(requested, "0") != 0) ||
                (enable && (!candidate->femLeftRowEquilibrationEnabled ||
                 !candidate->femCoupledRowEquilibrationEnabled ||
                 !candidate->femFullCoupledTrueResidualEnabled))) {
                diagnostics.message =
                    "square-root row scaling requires the admitted coupled row equilibration with a fresh physical certificate";
                return diagnostics;
            }
            candidate->femTemperedRowEquilibrationEnabled = enable;
            if (enable) {
                candidate->executionFingerprint = mixFingerprint(
                    candidate->executionFingerprint, 0x524f575351525431ull);
                std::fprintf(stderr, "matter_fgmres_row_scale_power=0.5 physical_certificate=unscaled\\n");
            }
        }
""" + insert_before)
    return replace(text, """                    [encoder setBuffer:state.femLeftRowEquilibrationScale
                                 offset:0u atIndex:2u];
                }, WorkDomain::femNodes);""", """                    [encoder setBuffer:state.femLeftRowEquilibrationScale
                                 offset:0u atIndex:2u];
                    const std::uint32_t tempered =
                        state.femTemperedRowEquilibrationEnabled ? 1u : 0u;
                    [encoder setBytes:&tempered length:sizeof(tempered) atIndex:3u];
                }, WorkDomain::femNodes);""")

def rewrite_shader(text):
    text = replace(text, """    device float* rowScale [[buffer(2)]],
    uint global""", """    device float* rowScale [[buffer(2)]],
    constant uint& tempered [[buffer(3)]],
    uint global""")
    return replace(text, "    rowScale[global] = scale;",
        "    rowScale[global] = tempered != 0u ? sqrt(scale) : scale;")

def main():
    shutil.copytree(SOURCE / "src", BUILD / "src", dirs_exist_ok=True)
    for name in ["runtime.mm", "app-runtime.mm"]:
        source = (SOURCE / name).read_text()
        write_change(name, source, rewrite_runtime(source), "matter/src/runtime.mm")
    name = "src/metal/fgmres.metalinc"
    source = (SOURCE / name).read_text()
    write_change(name, source, rewrite_shader(source), "matter/src/metal/fgmres.metalinc")
    commands = json.loads((PREVIOUS / "row-transform-build-commands.json").read_text())
    commands = [[arg.replace(str(SOURCE), str(BUILD)) for arg in cmd] for cmd in commands]
    (HERE / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    for cmd in commands:
        print(cmd[0], cmd[-1], flush=True)
        subprocess.run(cmd, cwd=Path.home() / "Work/skin-fat-native-integration-209/build-native",
                       check=True)

if __name__ == "__main__":
    main()
