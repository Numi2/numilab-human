"""Build a paired native test for the two interior skin/fat reference edits."""
from argparse import ArgumentParser
import csv
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
COARSE = HERE.parents[1]
BASE = COARSE / "exact-shell-bone-audit-032"
REFERENCE = COARSE / "thickness-rebuild-034/compiled-reference"
BUILD = Path.home() / "Work/skin-fat-native-integration-209/build-native"
APP = "numilab_human_myosim_visual_probe.mm"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inject(source, reference, fit, fit_sha256):
    nodes = np.flatnonzero(np.any(reference != fit, axis=1))
    assert nodes.tolist() == [3703, 4034]
    moves = []
    for node in nodes:
        row = node - 2411
        point = ", ".join(map(str, reference[node]))
        delta = fit[node] - reference[node]
        assert np.linalg.norm(delta) < .00055
        shift = ", ".join(map(str, delta))
        moves.append(f"                {{{row}u, {{{point}}}, {{{shift}}}}}")
    layer_mark = "        const double skinDebit=enableWholeBodySkinFat?kWholeBodySkinMassDebitKg:"
    hash_mark = "        const auto digest=loadedKneeSHA256(identityText.data(),identityText.size());"
    assert source.count(layer_mark) == source.count(hash_mark) == 1
    block = r'''        const char* innerFitSetting=std::getenv("NUMI_HUMAN_LOAD_WITNESS_INNER_FIT");
        if(innerFitSetting) {
            struct InteriorMove {std::size_t row; V3 expected; V3 delta;};
            constexpr std::array<InteriorMove,2> moves{{
POINTS
            }};
            require(std::string_view(innerFitSetting)=="1"&&enableWholeBodySkinFat&&
                        handClearanceSetting!=nullptr&&!registeredVolumeInstallRequested&&
                        region.worldPoints.size()==2411u,
                    "load witness trial requires the matched coarse shell");
            for(const auto& move:moves) {
                const auto row=move.row;
                require(norm(sub(skinLayer.innerWorld[row],move.expected))<1.0e-6,
                        "load witness trial lost its pinned native reference row");
                const auto& frame=region.worldToDonorLocalLinear[row];
                V3 local{};
                for(unsigned axis=0;axis<3;++axis)
                    for(unsigned lane=0;lane<3;++lane)
                        local[axis]+=frame[3u*axis+lane]*move.delta[lane];
                skinLayer.innerWorld[row]=add(skinLayer.innerWorld[row],move.delta);
                skinLayer.innerLocal[row]=add(skinLayer.innerLocal[row],local);
                fatLayer.outerWorld[row]=skinLayer.innerWorld[row];
                fatLayer.outerLocal[row]=skinLayer.innerLocal[row];
                skinThickness[row]=norm(sub(skinLayer.outerWorld[row],skinLayer.innerWorld[row]));
                fatThickness[row]=norm(sub(fatLayer.outerWorld[row],fatLayer.innerWorld[row]));
                require(skinThickness[row]>=reference.minSkinThicknessM&&
                            skinThickness[row]<=reference.maxSkinThicknessM&&
                            fatThickness[row]>=reference.minFatThicknessM&&
                            fatThickness[row]<=reference.maxFatThicknessM,
                        "load witness trial violates the original layer thickness bounds");
                skinLayer.thicknessByVertexM[row]=skinThickness[row];
                fatLayer.thicknessByVertexM[row]=fatThickness[row];
            }
            const auto skinRange=std::minmax_element(skinThickness.begin(),skinThickness.end());
            const auto fatRange=std::minmax_element(fatThickness.begin(),fatThickness.end());
            skinLayer.minThicknessM=*skinRange.first;
            skinLayer.maxThicknessM=*skinRange.second;
            fatLayer.minThicknessM=*fatRange.first;
            fatLayer.maxThicknessM=*fatRange.second;
            std::cout<<"load_witness_inner_fit=installed interior_nodes=2 candidate_sha256=FIT_HASH\n";
        }
'''
    block = block.replace("POINTS", ",\n".join(moves)).replace("FIT_HASH", fit_sha256)
    source = source.replace(layer_mark, block + layer_mark)
    return source.replace(hash_mark,
        f'        if(innerFitSetting)identityText+="|load-witness-inner-fit-v1|{fit_sha256}";\n' + hash_mark)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fit", type=Path, default=HERE.parent / "fit-exact-21/candidate.npz")
    args = parser.parse_args()
    source = (BASE / APP).read_text()
    fit = args.fit.resolve()
    receipt = json.loads((fit.parent / "receipt.json").read_text())
    assert digest(fit) == receipt["candidate_sha256"]
    node_path = REFERENCE / "shell-mass.csv.nodes.csv"
    assert digest(node_path) == receipt["input_sha256"][str(node_path)]
    with node_path.open() as stream:
        reference = np.array([[float(row[f"rest_{axis}_m"]) for axis in "xyz"]
                              for row in csv.DictReader(stream)])
    with np.load(fit, allow_pickle=False) as data:
        candidate = inject(source, reference, data["rest"], digest(fit))
    output = HERE / APP
    output.write_text(candidate)
    difference = difflib.unified_diff(source.splitlines(True), candidate.splitlines(True),
                                     f"a/apps/{APP}", f"b/apps/{APP}")
    (HERE / "inner-fit.patch").write_text("".join(difference))
    commands = [[str(HERE / Path(a).name) if Path(a).parent == BASE else a for a in argv]
                for argv in json.loads((BASE / "build-commands.json").read_text())]
    (HERE / "build-pins.json").write_text(json.dumps({"commands": commands, "base_app_sha256": digest(BASE / APP),
        "reference_sha256": digest(node_path), "candidate_sha256": digest(fit), "updated_app_sha256": digest(output)},
        indent=2) + "\n")
    if args.compile:
        for argv in commands:
            subprocess.run(argv, cwd=BUILD, check=True)


if __name__ == "__main__":
    main()
