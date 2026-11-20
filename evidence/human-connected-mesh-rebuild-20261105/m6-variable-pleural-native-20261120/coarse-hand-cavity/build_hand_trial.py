"""Compile the isolated, pinned i-layer trial in the existing Human scene."""
import difflib
import hashlib
import json
import math
import sys
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "resting-clearance-and-atrium-017"
SOURCE = SCENE / "floor-scene-build"
BASE = HERE.parent / "resting-clearance-and-atrium-015/staged-solver-accuracy-build"
RUNTIME = SCENE / "guarded-runtime-admission-build"
OUT = HERE / "native-build"
CONNECTED_OUT = HERE / "connected-native-build"
WITNESS_OUT = HERE / "witness-native-build"
CLEARANCE_OUT = HERE / "clearance-native-build"
CWD = Path.home() / "Work/skin-fat-native-integration-209/build-native"
APP = "numilab_human_myosim_visual_probe.mm"
FIT = HERE / "hand-interface-fit.json"
CONNECTED_FIT = HERE / "connected-interface-fit.json"
CLEARANCE_FIT = HERE / "hand-clearance-fit.json"

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def inject(source, fit_path):
    fit = json.loads(fit_path.read_text())
    nodes = fit["nodes"] if "nodes" in fit else [fit]
    assert fit["source_layer"] == 1 and 1 <= len(nodes) <= 30
    entries = []
    for node in nodes:
        row = node["local_node"] - 2411
        reference = ", ".join(map(repr, node["reference_world_m"]))
        delta = ", ".join(map(repr, node["world_delta_m"]))
        assert 0 <= row < 2411
        entries.append(f"                    {{{row}u, {{{reference}}}, {{{delta}}}}}")
    assert len({node["local_node"] for node in nodes}) == len(nodes)
    debit = "        const double skinDebit=enableWholeBodySkinFat?kWholeBodySkinMassDebitKg:"
    assert source.count(debit) == 1
    block = r"""        const char* handFitSetting=std::getenv("NUMI_HUMAN_COARSE_HAND_IFACE_TRIAL");
        if(handFitSetting) {
            struct HandInterfacePoint {std::size_t row; V3 expectedWorld; V3 shift;};
            constexpr std::array<HandInterfacePoint, COUNT> fit{{
ENTRIES
            }};
            require(std::string_view(handFitSetting)=="1"&&enableWholeBodySkinFat&&
                    !registeredVolumeInstallRequested&&region.worldPoints.size()==2411u,
                    "hand interface trial requires the exact diagnostic coarse parent");
            for(const auto& point:fit) {
                require(norm(sub(skinLayer.innerWorld[point.row],point.expectedWorld))<=2.0e-10,
                        "hand interface trial donor world row differs from its diagnostic source");
            }
            double maxShift=0.;
            for(const auto& point:fit) {
                const auto row=point.row;
                const auto shift=point.shift;
                const auto& frame=region.worldToDonorLocalLinear[row];
                V3 localShift{};
                for(unsigned i=0;i<3u;++i)
                    for(unsigned j=0;j<3u;++j)localShift[i]+=frame[3u*i+j]*shift[j];
                skinLayer.innerWorld[row]=add(skinLayer.innerWorld[row],shift);
                skinLayer.innerLocal[row]=add(skinLayer.innerLocal[row],localShift);
                fatLayer.outerWorld[row]=skinLayer.innerWorld[row];
                fatLayer.outerLocal[row]=skinLayer.innerLocal[row];
                skinThickness[row]=norm(sub(skinLayer.outerWorld[row],skinLayer.innerWorld[row]));
                fatThickness[row]=norm(sub(fatLayer.outerWorld[row],fatLayer.innerWorld[row]));
                skinLayer.thicknessByVertexM[row]=skinThickness[row];
                fatLayer.thicknessByVertexM[row]=fatThickness[row];
                maxShift=std::max(maxShift,norm(shift));
            }
            const auto s=std::minmax_element(skinThickness.begin(),skinThickness.end());
            const auto f=std::minmax_element(fatThickness.begin(),fatThickness.end());
            skinLayer.minThicknessM=*s.first;skinLayer.maxThicknessM=*s.second;
            fatLayer.minThicknessM=*f.first;fatLayer.maxThicknessM=*f.second;
            std::cout<<"coarse_hand_interface_trial moved_nodes="<<fit.size()
                     <<" maximum_displacement_m="<<maxShift<<" fit_sha256=FIT_SHA\\n";
        }
""".replace("COUNT", str(len(entries))).replace("ENTRIES", ",\n".join(entries)).replace("FIT_SHA", digest(fit_path))
    before_hash = "        const auto digest=loadedKneeSHA256(identityText.data(),identityText.size());"
    assert source.count(before_hash) == 1
    source = source.replace(debit, block + debit)
    return source.replace(before_hash,
        '        if(handFitSetting)identityText+="|coarse-hand-interface-v1|' + digest(fit_path) +
        '";\n' + before_hash)

def inject_clearance(source, fit_path=CLEARANCE_FIT):
    fit = json.loads(fit_path.read_text())
    assert fit["tangent_plane_fit_pass"]
    nodes = fit["candidate_nodes"]
    assert 4 <= len(nodes) <= 3 * 2411
    assert len({node["local_node"] for node in nodes}) == len(nodes)
    entries = []
    for node in nodes:
        role, row = divmod(node["local_node"], 2411)
        assert 0 <= role < 3
        coordinates, shifts = node["source_world_m"], node["world_delta_m"]
        assert len(coordinates) == len(shifts) == 3
        assert all(math.isfinite(value) for value in coordinates + shifts)
        ref = ", ".join(map(repr, coordinates))
        delta = ", ".join(map(repr, shifts))
        entries.append(f"                   {{{role}u, {row}u, {{{ref}}}, {{{delta}}}}}")
    point = "        const double skinDebit=enableWholeBodySkinFat?kWholeBodySkinMassDebitKg:"
    digest_point = "        const auto digest=loadedKneeSHA256(identityText.data(),identityText.size());"
    fit_hash = digest(fit_path)
    assert source.count(point) == source.count(digest_point) == 1
    block = (HERE / "hand-clearance-trial.mm.inc").read_text()
    block = block.replace("POINT_COUNT", str(len(nodes))).replace("POINTS", ",\n".join(entries))
    block = block.replace("FIT_SHA", fit_hash)
    return source.replace(point, block + point).replace(digest_point,
        '        if(handClearanceSetting)identityText+="|coarse-hand-clearance-v1|' + fit_hash +
        '";\n' + digest_point)


def main():
    mode = sys.argv[1:]
    if mode and mode[0] == "--bank-audit":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("fit", type=Path)
        parser.add_argument("out", type=Path)
        options = parser.parse_args(mode[1:])
        out = options.out.resolve()
        out.mkdir(exist_ok=False)
        source = (SOURCE / APP).read_text()
        updated = inject_clearance(source, options.fit)
        spot = '        const char* fullBankComparisonPath=std::getenv('
        assert updated.count(spot) == 1
        updated = updated.replace(spot, (HERE / "bank-witness.mm.inc").read_text() + spot)
        build(source, updated, out, out / "native.patch")
        return
    assert mode in ([], ["--connected"], ["--witness"], ["--clearance"])
    connected = bool(mode)
    witness = mode == ["--witness"]
    clearance = mode == ["--clearance"]
    out = CLEARANCE_OUT if clearance else WITNESS_OUT if witness else CONNECTED_OUT if connected else OUT
    fit_path = CONNECTED_FIT if connected else FIT
    out.mkdir(exist_ok=True)
    source = (SOURCE / APP).read_text()
    updated = inject_clearance(source) if clearance else inject(source, fit_path)
    if witness or clearance:
        spot = '        const char* fullBankComparisonPath=std::getenv('
        assert updated.count(spot) == 1
        updated = updated.replace(spot, (HERE / "hand-solid-witness.mm.inc").read_text() + spot)
    patch_name = ("hand-clearance-native.patch" if clearance else "hand-solid-witness.patch" if witness else "connected-interface-native.patch"
                  if connected else "hand-interface-native.patch")
    build(source, updated, out, HERE / patch_name)


def build(source, updated, out, patch_path):
    (out / APP).write_text(updated)
    patch = difflib.unified_diff(source.splitlines(keepends=True), updated.splitlines(keepends=True),
                                fromfile="a/apps/" + APP, tofile="b/apps/" + APP)
    patch_path.write_text("".join(patch))
    compile_app, link_app = json.loads((BASE / "build-commands.json").read_text())
    names = {APP, "numi-human-native.o", "numi-human-native"}
    compile_app = [str(out / Path(arg).name)
                   if Path(arg).parent == BASE and Path(arg).name in names else arg for arg in compile_app]
    compile_app[1:1] = ["-iquote", str(SOURCE), "-iquote", str(BASE)]
    link_app = [str(RUNTIME / "app-runtime.mm.o") if arg.endswith("/runtime.mm.o")
                else str(out / Path(arg).name) if Path(arg).parent == BASE and Path(arg).name in names
                else str(SOURCE / Path(arg).name) if Path(arg).parent == BASE
                else arg for arg in link_app]
    commands = [compile_app, link_app]
    (out / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    for command in commands:
        subprocess.run(command, cwd=CWD, check=True)
    for shader in SOURCE.glob("*.metallib"):
        target = out / shader.name
        if not target.exists():
            target.symlink_to(shader)
    print("Native binary SHA-256:", digest(out / "numi-human-native"), flush=True)

if __name__ == "__main__":
    main()
