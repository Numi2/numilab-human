"""Compile a fixed-boundary cavity through the existing native paired-layer owner."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from cavity import determinant
from map_cavity import digest, overlap
from native_trial import array, once, render_header


def table(text, name, width=1):
    body = re.search(r"kElbowRefinement" + name + r"=\{\{(.*?)\}\};", text, re.S)[1]
    values = np.array([int(v) for v in re.findall(r"(\d+)u", body)])
    return values.reshape(-1, width) if width > 1 else values


CANDIDATE = r"""
            if (posteriorBedReference) {
                cavityChildren.assign(kElbowRefinementChildTetrahedra.begin(),kElbowRefinementChildTetrahedra.end());
                cavityParents.assign(kElbowRefinementChildParentTetrahedra.begin(),kElbowRefinementChildParentTetrahedra.end());
                cavityFaces.assign(kElbowRefinementChildParentSurfaceFaces.begin(),kElbowRefinementChildParentSurfaceFaces.end());
                cavityStableFaces.assign(kElbowRefinementChildSourceFaceIdentities.begin(),kElbowRefinementChildSourceFaceIdentities.end());
                constexpr auto invalid=std::numeric_limits<std::uint32_t>::max();
                cavityParentStable.assign(kElbowRefinementParentTetCount,invalid);
                for(std::size_t child=0;child<cavityChildren.size();++child) {
                    auto& stable=cavityParentStable[cavityParents[child]];
                    require(stable==invalid || stable==cavityStableFaces[child],
                            "posterior source children disagree on parent stable-face identity");
                    stable=cavityStableFaces[child];
                }
                require(std::none_of(cavityParentStable.begin(),cavityParentStable.end(),
                                     [=](auto face){return face==invalid;}),
                        "posterior source parent lacks a retained stable-face identity");
                cavityOffsets.reserve(cavityChildren.size()+1u);
                cavityAncestry.reserve(cavityChildren.size()+4u*kPosteriorCavityRows.size());
                cavityFractions.reserve(cavityAncestry.capacity());
                std::size_t update=0;
                for(std::size_t child=0;child<cavityChildren.size();++child) {
                    cavityOffsets.push_back(static_cast<std::uint32_t>(cavityAncestry.size()));
                    if(update<kPosteriorCavityRows.size() && child==kPosteriorCavityRows[update]) {
                        auto original=cavityChildren[child];
                        std::sort(original.begin(),original.end());
                        require(original==kPosteriorCavityOriginal[update],
                                "posterior cavity no longer matches source");
                        cavityChildren[child]=kPosteriorCavityReplacement[update];
                        cavityParents[child]=kPosteriorCavityDominant[update];
                        cavityFaces[child]=kPosteriorCavitySurfaceFaces[update];
                        cavityStableFaces[child]=kPosteriorCavityStableFaces[update];
                        for(std::size_t entry=0;entry<kPosteriorCavityParentCandidates.size();++entry) {
                            const double fraction=kPosteriorCavityVolumeFractions[update][entry];
                            if(fraction>0.) {
                                cavityAncestry.push_back(kPosteriorCavityParentCandidates[entry]);
                                cavityFractions.push_back(fraction);
                            }
                        }
                        ++update;
                    } else {
                        cavityAncestry.push_back(cavityParents[child]);
                        cavityFractions.push_back(1.);
                    }
                }
                require(update==kPosteriorCavityRows.size(),"posterior cavity edit is incomplete");
                cavityOffsets.push_back(static_cast<std::uint32_t>(cavityAncestry.size()));
                std::map<std::array<std::uint32_t,3>,std::uint32_t> registeredFaces;
                for(std::size_t face=0;face<region.faces.size();++face) {
                    auto key=region.faces[face];std::sort(key.begin(),key.end());
                    require(registeredFaces.emplace(key,static_cast<std::uint32_t>(face)).second,
                            "posterior cavity has a duplicate registered layer face");
                }
                cavityFacetFaces.assign(4u*cavityChildren.size(),invalid);
                cavityFacetStable.assign(cavityFacetFaces.size(),invalid);
                const auto nodes=static_cast<std::uint32_t>(region.worldPoints.size());
                for(std::size_t child=0;child<cavityChildren.size();++child) {
                    for(std::size_t omitted=0;omitted<4u;++omitted) {
                        std::array<std::uint32_t,3> key{};
                        std::size_t at=0;
                        for(std::size_t corner=0;corner<4u;++corner)
                            if(corner!=omitted)key[at++]=cavityChildren[child][corner];
                        const auto layer=key[0]/nodes;
                        if(key[1]/nodes!=layer||key[2]/nodes!=layer)continue;
                        for(auto& index:key)index%=nodes;
                        std::sort(key.begin(),key.end());
                        const auto match=registeredFaces.find(key);
                        if(match==registeredFaces.end())continue;
                        const auto face=match->second;
                        cavityFacetFaces[4u*child+omitted]=face;
                        cavityFacetStable[4u*child+omitted]=region.sourceFaceIndices.empty()?face:region.sourceFaceIndices[face];
                    }
                }
                explicitTopology->childTetrahedra=cavityChildren;
                explicitTopology->childParentTetrahedra=cavityParents;
                explicitTopology->childParentSurfaceFaces=cavityFaces;
                explicitTopology->childSourceFaceIdentities=cavityStableFaces;
                explicitTopology->childParentAncestryOffsets=cavityOffsets;
                explicitTopology->childParentAncestryTetrahedra=cavityAncestry;
                explicitTopology->childParentAncestryVolumeFractions=cavityFractions;
                explicitTopology->parentSourceFaceIdentities=cavityParentStable;
                explicitTopology->childRegisteredLayerFacetSurfaceFaces=cavityFacetFaces;
                explicitTopology->childRegisteredLayerFacetSourceFaceIdentities=cavityFacetStable;
            }
"""


def bind_source(text, fit, header):
    parent = table(text, "ChildParentTetrahedra")
    old = table(text, "ChildTetrahedra", 4)
    rest, remesh, original, edited = (fit[key] for key in
                                      ("nodes_reference", "cells", "original_cells", "changed_rows"))
    assert np.array_equal(np.sort(old, axis=1), np.sort(original, axis=1))
    parents = np.unique(parent[edited])
    assert len(parents) == len(edited) == 4
    assert np.count_nonzero(np.isin(parent, parents)) == len(edited)
    source_tetrahedra = table(text, "ParentTetrahedra", 4)[parents]
    source_n = int(re.search(r"kElbowRefinementParentVertexCount=(\d+)u", text)[1])
    nodes_n = len(rest)//3
    geometric_previous = source_tetrahedra % source_n + (source_tetrahedra//source_n)*nodes_n
    equivalent = original[edited][np.argsort(parent[edited])]
    assert np.array_equal(np.sort(geometric_previous, axis=1), np.sort(equivalent, axis=1))
    cells = remesh[edited]
    points = rest[np.unique(cells)].astype(float)
    scale = float(np.ptp(points, axis=0).max())
    local = (rest - points.mean(axis=0)) / scale
    ref_overlap = np.array([[overlap(local[new], local[prior]) for prior in geometric_previous] for new in cells])
    before, after = (determinant(local, tets)/6. for tets in (geometric_previous, cells))
    np.testing.assert_allclose(ref_overlap.sum(axis=0), before, rtol=2.e-8, atol=1.e-13)
    np.testing.assert_allclose(ref_overlap.sum(axis=1), after, rtol=2.e-8, atol=1.e-13)
    fractions = ref_overlap / ref_overlap.sum(axis=1)[:, None]
    idx = edited[np.argsort(parent[edited])][np.argmax(fractions, axis=1)]

    def constants(name, entries, form):
        if entries.ndim == 2:
            return array(name, f"std::array<{form},4>", [
                "{" + ",".join(f"{value}u" if form != "double" else f"{value:.17g}" for value in row) + "}"
                for row in entries])
        return array(name, form, [f"{value}u" for value in entries])

    for tag, data in {
        "Rows": edited, "Original": np.sort(original[edited], axis=1), "Replacement": cells,
        "ParentCandidates": parents, "Dominant": parent[idx],
        "SurfaceFaces": table(text, "ChildParentSurfaceFaces")[idx],
        "StableFaces": table(text, "ChildSourceFaceIdentities")[idx],
    }.items():
        header += constants("kPosteriorCavity" + tag, data, "std::uint32_t")
    header += constants("kPosteriorCavityVolumeFractions", fractions, "double")
    return header


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("build", "reference", "remapped", "topology", "output"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    a = parser.parse_args()
    a.output.mkdir()
    fit = a.remapped / "posterior-interface-fit.npz"
    header = render_header(a.reference, a.remapped)
    with np.load(fit) as data:
        header = bind_source(a.topology.read_text(), data, header)
    (a.output / "PosteriorBedReference.inc").write_text(header)
    name = "numilab_human_myosim_visual_probe.mm"
    source = (a.build / name).read_text()
    existing = "        std::optional<ExplicitPairedLayerTopology> explicitTopology;"
    source = once(source, existing, existing + """
        std::vector<std::array<std::uint32_t,4>> cavityChildren;
        std::vector<std::uint32_t> cavityParents,cavityFaces,cavityStableFaces,cavityOffsets,cavityAncestry;
        std::vector<std::uint32_t> cavityFacetFaces,cavityFacetStable,cavityParentStable;
        std::vector<double> cavityFractions;""")
    close = '''                std::span<const std::uint32_t>(kElbowRefinementChildSourceFaceIdentities)});'''
    source = once(source, close, close + CANDIDATE)
    (a.output / name).write_text(source)
    commands = [[piece.replace(str(a.build), str(a.output)) for piece in command]
                for command in json.loads((a.build / "build-commands.json").read_text())]
    (a.output / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with (a.output / "build.log").open("xb") as out:
        for command in commands:
            subprocess.run(command, stdout=out, stderr=out, check=True, timeout=180)
    edited = (a.output / name, a.output / "PosteriorBedReference.inc", a.output / "numi-human-native")
    receipt = {p.name: digest(p) for p in (*edited, a.topology, fit)}
    (a.output / "build-identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt)


if __name__ == "__main__":
    main()
