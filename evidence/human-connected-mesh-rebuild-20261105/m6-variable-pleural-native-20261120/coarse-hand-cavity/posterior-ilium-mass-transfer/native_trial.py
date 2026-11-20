"""Compile the pinned posterior reference and mass ledger into the existing Human owner."""
from argparse import ArgumentParser
import difflib
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np

from audit import digest, rows


LAYER = r"""
        if(posteriorBedReference) {
            require(enableWholeBodySkinFat && handClearanceSetting && innerFitSetting &&
                    !registeredVolumeInstallRequested && region.worldPoints.size()==2411u,
                    "posterior bed reference requires the matched three-layer source");
            for(const auto& fit:kPosteriorDeepMoves) {
                const auto row=static_cast<std::size_t>(fit[0]);
                const V3 expected{fit[1],fit[2],fit[3]};
                require(norm(sub(fatLayer.innerWorld[row],expected))<1e-6,
                        "posterior fat source geometry differs from its verified fit");
                const V3 worldDelta{0.,0.,fit[4]};
                const auto& frame=region.worldToDonorLocalLinear[row];
                const V3 localDelta{frame[2]*fit[4],frame[5]*fit[4],frame[8]*fit[4]};
                fatLayer.innerWorld[row]=add(fatLayer.innerWorld[row],worldDelta);
                fatLayer.innerLocal[row]=add(fatLayer.innerLocal[row],localDelta);
                fatThickness[row]=norm(sub(fatLayer.innerWorld[row],fatLayer.outerWorld[row]));
                fatLayer.thicknessByVertexM[row]=fatThickness[row];
            }
            const auto range=std::minmax_element(fatThickness.begin(),fatThickness.end());
            fatLayer.minThicknessM=*range.first;
            fatLayer.maxThicknessM=*range.second;
            require(fatLayer.minThicknessM>=reference.minFatThicknessM &&
                    fatLayer.maxThicknessM<=kPosteriorMaximumFatThicknessM,
                    "posterior fat layer violates the source-derived fitted thickness bound");
            reference.maxFatThicknessM=std::max(reference.maxFatThicknessM,fatLayer.maxThicknessM);
            const auto n=region.worldPoints.size();
            const auto point=[&](std::uint32_t row)->const V3& {
                return row<n?region.worldPoints[row]:
                    (row<2*n?skinLayer.innerWorld[row-n]:fatLayer.innerWorld[row-2*n]);
            };
            std::array<double,2> volume{};
            for(std::size_t t=0;t<kElbowRefinementChildTetrahedra.size();++t) {
                const auto& cell=kElbowRefinementChildTetrahedra[t];
                const auto material=kElbowRefinementChildMaterialCodes[t];
                require(material==1u||material==2u,"posterior shell child material identity is invalid");
                volume[material-1u]+=tetVolume(point(cell[0]),point(cell[1]),point(cell[2]),point(cell[3]));
            }
            skinLayer.volumeM3=volume[0];
            fatLayer.volumeM3=volume[1];
            fatThicknessProvenance+="|registered posterior gluteal witnesses and pelvic clearance";
            fatLayer.thicknessProvenance=fatThicknessProvenance;
        }
"""

MASS = r"""
        if(posteriorBedReference) {
            require(!shell.mixedFEM && shell.tetrahedra.size()==kPosteriorTetMassKg.size() &&
                    preparedSkin.donorMassNodes.size()==shell.femNodes.size() &&
                    preparedFat.donorMassNodes.empty(),
                    "posterior mass ledger no longer matches the bonded shell owner");
            // Explicit cell masses own inertia. Restore the original fixed
            // constitutive parameters to retain the generated Metal bundle.
            auto restoreDensity=[&](std::uint32_t material,double density) {
                auto& parameters=source.materials[material].parameters;
                auto field=std::find_if(parameters.begin(),parameters.end(),
                                      [](const auto& p){return p.name=="density";});
                require(field!=parameters.end() && density>=field->lower && density<=field->upper,
                        "posterior shell lost its original constitutive density");
                field->defaultValue=density;
            };
            restoreDensity(skinMaterialIndex,skinDebit/kWholeBodySkinLayerVolumeM3);
            restoreDensity(fatMaterialIndex,fatDebit/kWholeBodyFatLayerVolumeM3);
            std::vector<double> nodalMass(shell.femNodes.size());
            std::array<double,2> layerMass{};
            for(std::size_t t=0;t<shell.tetrahedra.size();++t) {
                const auto& nodes=shell.tetrahedra[t].nodes;
                auto actual=nodes;
                std::sort(actual.begin(),actual.end());
                require(actual==kPosteriorTetNodeIdentities[t],
                        "posterior mass ledger no longer matches its source tetrahedron order");
                const auto material=shell.femMaterialIndices[t];
                require(material==skinMaterialIndex||material==fatMaterialIndex,
                        "posterior mass ledger encountered another material owner");
                const auto volume=std::abs(tetVolume(shell.femNodes[nodes[0]],shell.femNodes[nodes[1]],
                    shell.femNodes[nodes[2]],shell.femNodes[nodes[3]]));
                const auto mass=kPosteriorTetMassKg[t];
                const auto& parameters=source.materials[material].parameters;
                const auto density=std::find_if(parameters.begin(),parameters.end(),
                    [](const auto& param){return param.name=="density";});
                require(density!=parameters.end() && volume>0. && std::isfinite(mass) && mass>0.,
                        "posterior mass ledger needs a positive reference cell and material density");
                // The stored solver reference was rounded from these same
                // double coordinates. Keep the authored density bounds.
                require(mass/volume>=density->lower-1e-5 && mass/volume<=density->upper+1e-5,
                        "posterior mapped mass escapes its constitutive density limits");
                layerMass[material==fatMaterialIndex]+=mass;
                for(auto n:nodes)nodalMass[n]+=mass*.25;
            }
            // The mapper conserves original cooked masses. The legacy nominal
            // debits were computed before rounding reference nodes to Float32.
            constexpr double kSourceCookedMassAgreementKg=1e-6;
            require(std::abs(layerMass[0]-skinDebit)<=kSourceCookedMassAgreementKg &&
                    std::abs(layerMass[1]-fatDebit)<=kSourceCookedMassAgreementKg,
                    "posterior mass map disagrees with the nominal source tissue mass debit");
            for(auto& row:preparedSkin.donorMassNodes) {
                const std::size_t n=row.nodeIndex-preparedSkin.firstGlobalFEMNode;
                require(n<nodalMass.size() && nodalMass[n]>0.,
                        "posterior mass ledger does not cover every physical shell node");
                row.massKg=nodalMass[n];
            }
            shell.femTetrahedronMassesKg.assign(kPosteriorTetMassKg.begin(),kPosteriorTetMassKg.end());
            shell.femTetrahedronMassSourceIdentity=kPosteriorMassIdentity;
            std::cout<<"posterior_reference_mass_install=PASS deep_nodes="<<kPosteriorDeepMoves.size()
                     <<" tetrahedra="<<shell.femTetrahedronMassesKg.size()
                     <<" skin_kg="<<layerMass[0]<<" fat_kg="<<layerMass[1]
                     <<" source_sha256="<<kPosteriorMassSHA256<<std::endl;
        }
"""


def once(text, anchor, replacement):
    assert text.count(anchor) == 1, anchor
    return text.replace(anchor, replacement, 1)


def array(name, kind, entries):
    return f"static constexpr std::array<{kind},{len(entries)}> {name}{{{{\n" + ",\n".join(entries) + "\n}};\n"


def render_header(reference, fitted):
    r = rows(reference / "shell-mass.csv.nodes.csv")
    baseline = np.array([[v[f"rest_{a}_m"] for a in "xyz"] for v in r], dtype=float)
    with np.load(fitted / "posterior-interface-fit.npz") as candidate:
        positions = candidate["nodes_reference"]
        connectivity = candidate["cells"]
    mapped_mass = fitted / "balanced-mass.npz"
    with np.load(mapped_mass) as data:
        masses = data["cell_mass_kg"]
    n = len(baseline) // 3
    changes = np.flatnonzero(np.any(positions != baseline, axis=1))
    assert len(positions) == 3*n and np.all(changes >= 2*n)
    delta = positions - baseline
    assert not np.any(delta[:, :2])
    max_thickness = np.linalg.norm(positions[n:2*n] - positions[2*n:], axis=1).max()
    assert max_thickness < .035
    total = digest(mapped_mass)
    h = f'constexpr auto kPosteriorMassSHA256="{total}";\n'
    h += f'constexpr auto kPosteriorFitSHA256="{digest(fitted / "posterior-interface-fit.npz")}";\n'
    h += f'constexpr double kPosteriorMaximumFatThicknessM={float(max_thickness) + 1e-7:.17g};\n'
    h += array("kPosteriorMassIdentity", "std::uint64_t",
               [f"0x{total[i:i+16]}ull" for i in range(0, 64, 16)])
    h += array("kPosteriorDeepMoves", "std::array<double,5>",
               ["{" + ",".join(f"{v:.17g}" for v in (i-2*n, *baseline[i], delta[i, 2])) + "}"
                for i in changes])
    h += array("kPosteriorTetMassKg", "double", [f"{m:.17g}" for m in masses])
    h += array("kPosteriorTetNodeIdentities", "std::array<std::uint32_t,4>",
               ["{" + ",".join(f"{n}u" for n in sorted(nodes)) + "}" for nodes in connectivity])
    return h


def compile_owner(base, reference, fitted, out):
    out.mkdir()
    header = out / "PosteriorBedReference.inc"
    header.write_text(render_header(reference, fitted))
    name = "numilab_human_myosim_visual_probe.mm"
    text = (base / name).read_text()
    patched = once(text, "        const double skinDebit=enableWholeBodySkinFat?",
                   '        const bool posteriorBedReference=std::getenv("NUMI_HUMAN_POSTERIOR_BED_REFERENCE")!=nullptr;\n'
                   '        #include "PosteriorBedReference.inc"\n'
                   + LAYER + "        const double skinDebit=enableWholeBodySkinFat?")
    patched = once(patched, "        const std::string fatThicknessProvenance=",
                   "        std::string fatThicknessProvenance=")
    patched = once(patched, "        const auto digest=loadedKneeSHA256(identityText.data(),identityText.size());",
                   '        if(posteriorBedReference){identityText+="|posterior-bed-reference-v1|";'
                   'identityText+=kPosteriorFitSHA256;identityText+=kPosteriorMassSHA256;}\n'
                   "        const auto digest=loadedKneeSHA256(identityText.data(),identityText.size());")
    # The reference frame and donor bookkeeping consume the same cooked shell
    # as the native constitutive/bed contact owner.
    patched = once(patched, "        shell.femReferenceNodes=shell.femNodes;",
                   MASS + "        shell.femReferenceNodes=shell.femNodes;")
    (out / name).write_text(patched)
    (out / "native-posterior.patch").write_text("".join(difflib.unified_diff(
        text.splitlines(keepends=True), patched.splitlines(keepends=True), f"a/{name}", f"b/{name}")))
    baseline = json.loads((base / "build-pins.json").read_text())
    commands = [[arg.replace(str(base), str(out)) for arg in cmd] for cmd in baseline["commands"]]
    link = commands[-1]
    runtime_lib = Path(next(option.split(",", 2)[2] for option in link if option.startswith("-Wl,-rpath,")))
    link[link.index("lib/libmetalrobo.dylib")] = str(runtime_lib / "libmetalrobo.dylib")
    (out / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with (out / "build.log").open("xb") as log:
        for cmd in commands:
            subprocess.run(cmd, stdout=log, stderr=log, check=True, timeout=180)


def run_owner(reference, build, out, steps, cache=None):
    out.mkdir()
    source = reference / "launch.json"
    launch = json.loads(source.read_text())
    launch["argv"][0] = str(build / "numi-human-native")
    launch["argv"][4] = str(out / "native")
    launch["argv"][launch["argv"].index("--muscle-step-count")+1] = str(steps)
    launch["native_binary_sha256"] = digest(Path(launch["argv"][0]))
    launch["parent_launch_sha256"] = digest(source)
    launch["variant"] = "registered-posterior-reference-and-typed-mass"
    env = launch["environment"]
    for key in ("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE", "NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"):
        env.pop(key, None)
    env.update({
        "NUMI_HUMAN_LOAD_WITNESS_INNER_FIT": "1",
        "NUMI_HUMAN_POSTERIOR_BED_REFERENCE": "1",
        "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": str(out / "initial-skin-fem.csv"),
        "NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV": str(out / "shell-mass.csv"),
    })
    if cache is not None:
        record = json.loads((cache / "launch.json").read_text())
        receipt = json.loads((cache / "execution.json").read_text())
        assert receipt["returncode"] == 0
        assert record["native_binary_sha256"] == launch["native_binary_sha256"]
        assert record["variant"] == launch["variant"]
        pack = cache / "native/resting-human.mrvpack"
        env["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"] = str(pack)
        env["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"] = digest(pack)
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    inherited = {k:v for k,v in os.environ.items() if not k.startswith(("NUMI_", "NM_"))}
    start = monotonic()
    with (out / "stdout.log").open("xb") as log, (out / "stderr.log").open("xb") as err:
        result = subprocess.run(launch["argv"], cwd=launch["cwd"], env=inherited | env,
                                stdout=log, stderr=err, check=False, timeout=240)
    receipt = {"returncode": result.returncode, "wall_seconds": monotonic() - start}
    (out / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return result.returncode


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    c = commands.add_parser("build")
    for name in ("base", "reference", "fitted", "out"):
        c.add_argument(name, type=lambda v: Path(v).resolve())
    r = commands.add_parser("run")
    for name in ("reference", "build", "out"):
        r.add_argument(name, type=lambda v: Path(v).resolve())
    r.add_argument("--steps", type=int, default=4)
    r.add_argument("--cache", type=lambda v: Path(v).resolve())
    a = vars(p.parse_args())
    if a.pop("action") == "build":
        compile_owner(**a)
    else:
        raise SystemExit(run_owner(**a))
