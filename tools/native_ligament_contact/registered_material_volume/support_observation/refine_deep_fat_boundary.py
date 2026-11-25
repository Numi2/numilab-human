"""Transfer the registered deep-fat surface onto a conforming edge/face refinement."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse


PREFIX = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nodal_basis(old_count, splits, split_fractions, insertions, face_weights):
    rows = {}
    def insert(parents, child, weights):
        parents = np.asarray(parents, dtype=np.int64)
        weights = np.asarray(weights, dtype=float)
        if child != old_count + len(rows) or np.any(parents < 0) or np.any(parents >= child):
            raise ValueError("Refinement is missing a parent or repeats a child")
        if (len(set(parents)) != len(parents) or not np.isfinite(weights).all()
                or np.min(weights) < 0 or abs(sum(weights) - 1) > 1e-12):
            raise ValueError("Invalid parent simplex or weights")
        terms = {}
        for p, w in zip(parents, weights, strict=True):
            for root, u in rows.get(p, {p: 1.0}).items():
                terms[root] = terms.get(root, 0.0) + w * u
        if len(terms) > 3:
            raise ValueError("New boundary node crosses its parent face")
        rows[child] = terms
    for (u, v, child), t in zip(splits, split_fractions, strict=True):
        insert(np.array([u, v]), child, np.array([1-t, t]))
    for ids, weights in zip(insertions, face_weights, strict=True):
        insert(ids[:-1], ids[-1], weights)
    return rows


def face_key(face):
    return tuple(sorted(face))


def bind(launch, refined, out):
    cfg = json.loads(launch.read_bytes())
    env = cfg["environment"]
    def source(key):
        p = Path(env[PREFIX + key])
        if digest(p) != cfg["pins"][str(p)]:
            raise ValueError("Changed registered source " + key)
        return p
    def table(key, columns, dtype):
        return np.loadtxt(source(key + "_CSV"), skiprows=1, delimiter=",",
                          usecols=columns, dtype=dtype)
    def tetgen(path):
        return np.loadtxt(path, skiprows=1, usecols=(1, 2, 3))
    old_path = source("TETGEN_NODE")
    x = tetgen(old_path)
    X = tetgen(refined/"mesh.node")
    if not np.array_equal(X[:len(x)], x) or not np.array_equal(X, X.astype("f4")):
        raise ValueError("Refinement changed source nodes or lacks exact native coordinates")
    with np.load(refined/"source-overlay.npz") as ov:
        basis = nodal_basis(len(x), ov["node_splits"], ov["node_split_fractions"],
                            ov["node_face_insertions"], ov["node_face_weights"])
        face_parent = ov["parent_wall_D"]
    old_wall = np.load(old_path.parent/"material-walls.npz")["D"]
    new_wall = np.load(refined/"material-walls.npz")["D"]
    if (len(basis) != len(X)-len(x) or len(new_wall) != len(face_parent) or
            not np.array_equal(np.unique(face_parent), np.arange(len(old_wall)))):
        raise ValueError("Refinement is missing a node or boundary face")
    joint = table("COMBINED_NODE_MAP", (0, 7), int)
    row = np.full(len(X), -1, dtype=int)
    deep = np.unique(old_wall)
    selected = joint[np.isin(joint[:, 0]-1, deep)]
    if len(selected) != len(deep) or np.any(selected[:, 1] < 0):
        raise ValueError("Deep nodes must have unique material ownership")
    row[selected[:, 0]-1] = selected[:, 1]
    if not np.array_equal(np.sort(row[deep]), np.arange(len(deep))):
        raise ValueError("Deep surface registration is incomplete")
    row[len(x):] = np.arange(len(deep), len(deep) + len(basis))
    old_role_face = table("SURFACE_D_FACES", (1, 2, 3, 4, 5, 6), int)
    by_key = {face_key(f): i for i, f in enumerate(old_role_face[:, :3])}
    surface_parent = np.array([by_key[face_key(f)] for f in row[old_wall]])
    if not np.array_equal(np.sort(surface_parent), np.arange(len(old_wall))):
        raise ValueError("Role faces and material walls differ")
    parent = surface_parent[face_parent]
    role_face = row[new_wall].copy()
    roles = np.unique(new_wall)
    order = roles[np.argsort(row[roles])]
    pts = X[order]
    i, j, w = [], [], []
    deep_set = set(deep)
    role_set = set(roles)
    for child, terms in basis.items():
        if not deep_set.issuperset(terms) or child not in role_set:
            raise ValueError("Deep-face ancestry escapes its role")
        i.extend([row[child]-len(deep)]*len(terms))
        j.extend(row[list(terms)])
        w.extend(terms.values())
    P = sparse.csr_matrix((w, (i, j)), shape=(len(basis), len(deep)))
    coordinate_error = np.linalg.norm(P @ pts[:len(deep)] - pts[len(deep):], axis=1)
    # The stored split is evaluated at a representable Float32 material location.
    # Tie the tolerance to that row's original coordinates.
    ulp = np.abs(np.spacing((np.abs(P) @ np.abs(pts[:len(deep)])).astype("f4")))
    if np.any(coordinate_error > 2*np.linalg.norm(ulp, axis=1)):
        raise ValueError("Child point does not match its physical parent interpolation")
    def area_vectors(tri):
        v = pts[tri]
        return np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])*.5
    base_area = area_vectors(old_role_face[:, :3])
    aa = area_vectors(role_face)
    orientation = np.einsum("ij,ij->i", aa, base_area[parent])
    if np.any(orientation == 0):
        raise ValueError("Degenerate deep surface child")
    flip = orientation < 0
    role_face[flip] = role_face[flip][:, [0, 2, 1]]
    aa = area_vectors(role_face)
    summed = np.zeros_like(base_area)
    np.add.at(summed, parent, aa)
    area_err = np.linalg.norm(summed-base_area, axis=1)/np.linalg.norm(base_area, axis=1)
    scalar_area = np.bincount(parent,weights=np.linalg.norm(aa,axis=1))
    area_size = np.linalg.norm(base_area,axis=1)
    area_sum_error = abs(scalar_area-area_size)/area_size
    if np.any(area_err > 2e-5) or np.any(area_sum_error > 2e-5):
        raise ValueError("Refined oriented area fails to reproduce the parent")
    # Preserve original role face IDs on one child per source, appending the rest.
    _, first = np.unique(parent, return_index=True)
    extra = np.ones(len(parent), bool)
    extra[first] = False
    perm = np.r_[first, np.flatnonzero(extra)]
    faces = role_face[perm]
    parents = parent[perm]
    np.testing.assert_array_equal(parents[:len(old_role_face)], np.arange(len(old_role_face)))
    out.mkdir(parents=True, exist_ok=True)
    file = source("SURFACE_D_FACES_CSV")
    with file.open() as src, (out/"surface_d_faces.csv").open("w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator="\n")
        writer.writerow(next(reader))
        ancestry = [r[4:] for r in reader]
        writer.writerows([k, *f, *ancestry[p]] for k, (f,p) in enumerate(zip(faces,parents)))
    old_xyz = table("SURFACE_D_NODES", (2,3,4), float)
    if not np.array_equal(old_xyz, pts[:len(deep)]):
        raise ValueError("Native surface registration uses different coordinates")
    normal = np.zeros_like(pts)
    for k in range(3):
        np.add.at(normal, role_face[:,k], aa)
    orig_n = table("SURFACE_D_NODES", (5,6,7), float)
    base_n = np.zeros_like(old_xyz)
    for k in range(3):
        np.add.at(base_n,old_role_face[:,k],base_area)
    # Orient against the existing native convention wherever the patch is touched.
    touched = np.unique(old_role_face[np.bincount(parent)>1,:3])
    signed = np.sign(np.einsum("ij,ij->i", orig_n, base_n))
    new_sign = P @ signed
    if np.any(abs(new_sign)<1-1e-10) or np.any(signed[touched]==0):
        raise ValueError("Contradictory outward normals on the refined patch")
    patch = np.r_[touched, np.arange(len(deep),len(pts))]
    if np.any(np.linalg.norm(normal[patch],axis=1)==0):
        raise ValueError("Missing surface normal")
    normal[patch] *= (np.r_[signed,new_sign][patch]/np.linalg.norm(normal[patch],axis=1))[:,None]
    for key, dest in [("ROLE_D_FULL86","role_d_full86.csv"),
                      ("ROLE_D_CONTACT_THICKNESS","role_d_contact_thickness.csv")]:
        f = source(key+"_CSV")
        needed = set(P.indices.tolist())
        with f.open() as src, (out/dest).open("w") as dst:
            dst.write(src.readline())
            values = {}
            for idx, line in enumerate(src):
                dst.write(line)
                if idx in needed:
                    nums = np.fromstring(line, sep=",")
                    if nums[0] != idx:
                        raise ValueError("Wrong row ID in " + key)
                    values[idx] = nums[1:]
            if idx+1 != len(deep):
                raise ValueError("Wrong source role length")
            for idx in range(P.shape[0]):
                a, b = P.indptr[idx:idx+2]
                result = P.data[a:b] @ np.stack([values[k] for k in P.indices[a:b]])
                if not np.isfinite(result).all() or result.min() < 0:
                    raise ValueError("Invalid derived physical field")
                if key.endswith("FULL86") and (len(result)!=86 or abs(result.sum()-1)>1e-6):
                    raise ValueError("Derived donor weights do not form a partition of unity")
                if key.endswith("THICKNESS") and np.any(result <= 0):
                    raise ValueError("Non-positive tissue thickness")
                dst.write(",".join([str(idx+len(deep)), *map(repr, result.tolist())])+"\n")
    np.savez_compressed(out/"deep-registration.npz", nodes=order, patch=patch,
                        normal=normal[patch], parents=parents, new_wall_order=perm,
                        indptr=P.indptr, indices=P.indices, weights=P.data)
    result = dict(source_launch_sha256=digest(launch), element_sha256=digest(refined/"mesh.ele"),
                  node_sha256=digest(refined/"mesh.node"), source_overlay_sha256=digest(refined/"source-overlay.npz"),
                  old_nodes=len(deep), new_nodes=len(pts), old_faces=len(old_wall),
                  new_faces=len(faces), refined_parent_faces=int(sum(np.bincount(parent)>1)),
                  max_position_error_m=float(coordinate_error.max()),
                  max_relative_oriented_area_error=float(area_err.max()),
                  max_relative_area_error=float(area_sum_error.max()),
                  outputs={p.name:digest(p) for p in (out/"surface_d_faces.csv", out/"role_d_full86.csv",
                                                       out/"role_d_contact_thickness.csv",out/"deep-registration.npz")})
    (out/"deep-boundary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("launch","refined","out"):
        parser.add_argument(name,type=Path)
    print(json.dumps(bind(**vars(parser.parse_args())),indent=2))
