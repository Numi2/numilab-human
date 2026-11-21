"""Build an explicit, bounded scalp edit on the verified native control."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from cavity import determinant
from map_cavity import digest, overlap
from native_trial import array, once
from build_native import table


def source_array(header, name, count):
    declaration = re.search(r"\b" + name + r"\s*=\s*\{(.*?)\};", header, re.S)[1]
    values = np.array([int(n) for n in re.findall(r"\b(\d+)u?\b", declaration)])
    return values.reshape(-1, count)


def constants(name, rows):
    rows = np.asarray(rows)
    if rows.ndim == 1:
        return array(name, "std::uint32_t", [f"{v}u" for v in rows])
    return array(name, f"std::array<std::uint32_t,{rows.shape[1]}>",
                 ["{" + ",".join(f"{v}u" for v in row) + "}" for row in rows])


def projected_overlap(points, old, new):
    # The changed quadrilateral is an embedded two-manifold. Project along
    # the average oriented normal, then compute exact polygon-area overlap
    # in that two-dimensional plane to retain both source parents.
    triangles = points[np.concatenate((old, new))]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal = normals[:2].sum(axis=0)
    normal /= np.linalg.norm(normal)
    along = triangles[0, 1]-triangles[0, 0]
    along -= (along @ normal)*normal
    along /= np.linalg.norm(along)
    across = np.cross(normal, along)
    projected = (triangles-triangles[0, 0]) @ np.array([along, across]).T

    def cross(a, b):
        return a[0]*b[1]-a[1]*b[0]

    def clipped_area(subject, clip):
        vertices = list(subject)
        for p, q in zip(clip, np.roll(clip,-1,axis=0), strict=True):
            old_vertices, vertices = vertices, []
            if not old_vertices:
                return 0.
            for start, end in zip(old_vertices, old_vertices[1:] + old_vertices[:1], strict=True):
                ds, de = cross(q-p,start-p), cross(q-p,end-p)
                if (ds >= 0.) != (de >= 0.):
                    vertices.append(start + (end-start)*ds/(ds-de))
                if de >= 0.:
                    vertices.append(end)
        return abs(sum(cross(p,q) for p,q in zip(vertices,vertices[1:]+vertices[:1],strict=True)))*.5

    areas = np.array([[clipped_area(new_tri, old_tri) for old_tri in projected[:2]]
                      for new_tri in projected[2:]])
    original = np.array([cross(tri[1]-tri[0],tri[2]-tri[0])*.5 for tri in projected[:2]])
    replaced = np.array([cross(tri[1]-tri[0],tri[2]-tri[0])*.5 for tri in projected[2:]])
    assert np.all(original > 0.) and np.all(replaced > 0.)
    np.testing.assert_allclose(areas.sum(axis=0),original,rtol=1e-10,atol=1e-14)
    np.testing.assert_allclose(areas.sum(axis=1),replaced,rtol=1e-10,atol=1e-14)
    return areas/replaced[:,None]


def render(args):
    base = args.base
    output = args.output
    candidate = args.candidate
    verdict = json.loads(args.verification.read_text())
    geometry = candidate / "geometry.json"
    deck = candidate / "candidate.npz"
    assert verdict["input_sha256"][geometry.name] == digest(geometry)
    assert verdict["input_sha256"][deck.name] == digest(deck)
    choices = json.loads(geometry.read_text())["selected"]
    topology_text = args.topology.read_text()
    surface_text = args.surface.read_text()
    faces = source_array(surface_text,"kWholeBodySkinFaces",3)
    face_source = source_array(surface_text,"kWholeBodySkinSourceFaces",1).ravel()
    face_ids = {tuple(sorted(row)): i for i,row in enumerate(faces)}
    assert len(face_ids) == len(faces)
    base_children = table(topology_text,"ChildTetrahedra",4)
    base_parent = table(topology_text,"ParentTetrahedra",4)
    base_parent_of = table(topology_text,"ChildParentTetrahedra")
    base_child_face = table(topology_text,"ChildParentSurfaceFaces")
    base_child_stable = table(topology_text,"ChildSourceFaceIdentities")
    n0 = int(re.search(r"kElbowRefinementParentVertexCount=(\d+)u",topology_text)[1])
    n = 2411
    from audit import rows
    points = np.array([[r[f"rest_{axis}_m"] for axis in "xyz"] for r in rows(args.nodes)],float)
    assert len(points) == 3*n
    assert verdict["input_sha256"][args.nodes.name] == digest(args.nodes)
    with np.load(deck) as data:
        old, new, changed, mass = (data[k] for k in
                              ("original_cells","cells","changed_rows","cell_mass_kg"))
    assert np.all(determinant(points,new[changed]) > 0.)
    assert (len(np.unique(base_parent_of[changed])) == len(changed)
            and np.isin(base_parent_of,base_parent_of[changed]).sum() == len(changed))
    np.testing.assert_array_equal(np.sort(old[changed],axis=1),np.sort(base_children[changed],axis=1))
    np.testing.assert_array_equal(np.sort(old[changed],axis=1),
                np.sort(base_parent[base_parent_of[changed]]%n0+base_parent[base_parent_of[changed]]//n0*n,axis=1))
    replaced_faces, face_provenance = [], []
    replacement_child_face = {}
    for choice in choices:
        old_faces,new_faces = (np.array(choice[k])[:2] for k in ("before","after"))
        slots = [face_ids[tuple(sorted(row))] for row in old_faces]
        def sign(f):
            i,j,k = map(int,f)
            return (-1)**(int(i>j)+int(i>k)+int(j>k))
        flip = sign(faces[slots[0]]) != sign(old_faces[0])
        assert flip == (sign(faces[slots[1]]) != sign(old_faces[1]))
        if flip:
            new_faces = new_faces[:,[0,2,1]]
            old_faces = old_faces[:,[0,2,1]]
        fractions = projected_overlap(points,old_faces,new_faces)
        pairing = np.array([0,1] if np.trace(fractions)>=np.trace(fractions[:,::-1]) else [1,0])
        for i, replacement in enumerate(new_faces):
            slot = slots[pairing[i]]
            key = tuple(sorted(replacement))
            replaced_faces.append([slot,*faces[slot],*replacement])
            for part in choice["cavities"]:
                for cell in part["retired"]:
                    if tuple(sorted(set(new[cell]%n))) == key:
                        replacement_child_face[cell] = slot
            face_provenance.append({"derived_face_row":int(slot),
                "old_registered_face_rows":list(map(int,slots)),
                "old_source_face_labels":face_source[slots].tolist(),
                "projected_area_fractions":fractions[i].tolist(),
                "assigned_representative_original_face":int(face_source[slot])})
        for part in choice["cavities"]:
            rows_ = np.asarray(part["retired"])
            # Untouched cells may reference these original donors only if
            # they share the exact same volume ancestry, which was excluded above.
            assert set(base_child_face[rows_]) == set(slots)
            np.testing.assert_array_equal(base_child_stable[rows_], face_source[base_child_face[rows_]])
            old_corner = points[np.unique(old[rows_])]
            scale = np.ptp(old_corner,axis=0).max()
            x = (points-old_corner.mean(axis=0))/scale
            part["original_overlap_reference_m3"] = [
                [float(overlap(x[a],x[b])*scale**3) for a in old[rows_]] for b in new[rows_]]
    assert set(replacement_child_face) == set(changed)
    parents = base_parent_of[changed]
    assert np.all(new[changed]%n < n0)
    child_faces = np.array([replacement_child_face[t] for t in changed])
    known_header = (base / "PosteriorBedReference.inc").read_text()
    for name, declaration in (
        ("kPosteriorTetMassKg",array("kPosteriorTetMassKg","double",[f"{v:.17g}" for v in mass])),
        ("kPosteriorTetNodeIdentities",constants("kPosteriorTetNodeIdentities",np.sort(new,axis=1))),
        ("kPosteriorMassIdentity",array("kPosteriorMassIdentity","std::uint64_t",
                    [f"0x{digest(deck)[i:i+16]}ull" for i in range(0,64,16)])),
    ):
        original = re.search(r"static constexpr [^\n]+ " + name + r"\{\{.*?\}\};",known_header,re.S)[0]
        known_header = once(known_header,original,declaration.rstrip())
    field = re.search(r'constexpr auto kPosteriorMassSHA256="[^"]+";',known_header)[0]
    known_header = once(known_header,field,f'constexpr auto kPosteriorMassSHA256="{digest(deck)}";')
    # Reuse the tested posterior overlap extension for its four interior
    # cells. Scalp parents instead encode the actual derived cavity geometry.
    payload = constants("kScalpFaceRows",replaced_faces)
    payload += constants("kScalpRows",changed)
    payload += constants("kScalpOriginal",np.sort(old[changed],axis=1))
    payload += constants("kScalpChildren",new[changed])
    payload += constants("kScalpParents",new[changed]%n+(new[changed]//n)*n0)
    payload += constants("kScalpChildFaces",child_faces)
    payload += f'constexpr auto kScalpReferenceSHA256="{digest(geometry)}";\n'
    (output/"PosteriorBedReference.inc").write_text(known_header + payload)
    (output/"registration-provenance.json").write_text(json.dumps({
        "source_geometry_sha256":digest(args.topology), "surface_atlas_sha256":digest(args.surface),
        "candidate_sha256":digest(deck), "edited_parent_rows":parents.tolist(),
        "source_faces":face_provenance,"volume_patches":choices,
    },indent=2,allow_nan=False)+"\n")
    name = "numilab_human_myosim_visual_probe.mm"
    original = (base/name).read_text()
    # The overlay is tightly scoped to the experimental reference owner.
    from patch import patch
    (output/name).write_text(patch(original, once))
    commands = [[part.replace(str(base),str(output)) for part in command]
                for command in json.loads((base/"build-commands.json").read_text())]
    (output/"build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
    with (output/"build.log").open("xb") as log:
        for argv in commands:
            subprocess.run(argv,stdout=log,stderr=log,check=True,timeout=200)
    build_hashes = {p.name:digest(p) for p in (output/name,output/"PosteriorBedReference.inc",
                    output/"numi-human-native",output/"registration-provenance.json",args.verification)}
    (output/"build-identity.json").write_text(json.dumps(build_hashes,indent=2)+"\n")
    print(build_hashes)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("base","candidate","verification","topology","surface","nodes","output"):
        parser.add_argument(key,type=lambda p:Path(p).resolve())
    a = parser.parse_args()
    a.output.mkdir()
    render(a)
