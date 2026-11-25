"""Rebuild the native-tested shoulder surface inside its source NHTISS5 scene."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Tissue:
    def __init__(self, path):
        data = path.read_bytes()
        self.header = data[:64]
        magic, abi, count, bindings, vertices, indices, _ = struct.unpack_from("<8s6I", data)
        require(magic == b"NHTISS4\0" and abi == 5, "Unexpected tissue payload")
        self.records = np.frombuffer(data, "<u4", count * 8, 64).reshape(-1, 8)
        start = 64 + self.records.nbytes
        self.bindings = data[start:start + bindings * 36]
        start += len(self.bindings)
        self.vertices = np.frombuffer(data, "<f4", vertices * 14, start).reshape(-1, 14)
        start += self.vertices.nbytes
        self.indices = np.frombuffer(data, "<u4", indices, start)
        require(start + self.indices.nbytes == len(data), "Truncated or unexpected trailing data")

    def surface(self, row):
        start, count, first, size = map(int, row[2:6])
        triangles = self.indices[first:first + size].astype(np.int64) - start
        require(len(triangles) == size and np.all((triangles >= 0) & (triangles < count)),
                "Triangle outside its anatomical bank")
        return self.vertices[start:start + count], triangles


def vertex_normals(x, faces):
    a, b, c = x[faces].transpose(1, 0, 2)
    normal = np.cross(b - a, c - a)
    sums = np.zeros_like(x)
    for column in faces.T:
        np.add.at(sums, column, normal)
    lengths = np.linalg.norm(sums, axis=1)
    sums[lengths > 0] /= lengths[lengths > 0, None]
    return sums


def build_staged(parent, donor_path, atlas, proposal, proof_path, output):
    proof = json.loads(proof_path.read_bytes())
    config = json.loads((parent / "launch.json").read_bytes())
    argv = config["argv"]
    source = Path(argv[argv.index("--soft-tissue-payload") + 1])
    receipt = Path(argv[argv.index("--resting-anatomy-receipt") + 1])
    for path, key in ((parent / "launch.json", "accepted_parent"), (source, "input"),
                      (receipt, "original_receipt"), (donor_path, "reference_tissue"),
                      (atlas, "source_member"), (proposal, "registration_proposal")):
        require(digest(path) == proof[key], f"Unmatched {key} input {path}")

    main, donor = Tissue(source), Tissue(donor_path)
    sid = int(proof["stable_id"])
    require(sid == 87, "Expected the anatomical right teres major")
    row, = donor.records[donor.records[:, 6] == sid]
    original, triangles = donor.surface(row)
    with np.load(proposal, allow_pickle=False) as p:
        ids, faces = p["raw_to_unique"], p["triangles"]
        x, y, pins = p["original_source"], p["posed_source"], p["pins"]
    atlas_points = np.array([line.split()[1:4] for line in atlas.read_text().splitlines()
                             if line.startswith("v ")], dtype=float)
    require(np.array_equal((atlas_points * .001).astype("<f4"), original[:, :3]),
            "Anatomical source coordinates differ")
    require(np.array_equal(original[:, :3], x[ids]) and np.array_equal(faces, ids[triangles].reshape(-1, 3)),
            "Registration does not belong to this atlas topology")
    require(np.array_equal(x[pins], y[pins]) and np.isfinite(y).all(), "Invalid attachment or coordinate")
    na, nb = vertex_normals(x, faces), vertex_normals(y, faces)
    rotation = np.column_stack((np.cross(na, nb), 1 + np.einsum("ij,ij->i", na, nb)))
    require(np.linalg.norm(rotation[np.unique(faces)], axis=1).min() > .1, "Reversed anatomical normal")
    patch = original.copy()
    patch[:, :3] = y[ids]
    patch[:, 3:6] = Rotation.from_quat(rotation[ids]).apply(patch[:, 3:6])

    records = main.records.copy()
    require(np.count_nonzero(records[:, 6] == sid) == 1, "Unresolved destination bank")
    vertices, indices = [], []
    vertex_count = index_count = 0
    for i, bank in enumerate(main.records):
        verts, faces = (patch, triangles) if bank[6] == sid else main.surface(bank)
        records[i, 2:6] = (vertex_count, len(verts), index_count, len(faces))
        vertices.append(verts.tobytes())
        indices.append((faces + vertex_count).astype("<u4").tobytes())
        vertex_count += len(verts)
        index_count += len(faces)
    header = bytearray(main.header)
    struct.pack_into("<II", header, 20, vertex_count, index_count)
    output.mkdir(exist_ok=False, parents=True)
    target = output / source.name
    with target.open("wb") as stream:
        for section in (header, records.tobytes(), main.bindings, *vertices, *indices):
            stream.write(section)
    require(digest(target) == proof["output"], "Tissue differs from the native replay")

    doc = json.loads(receipt.read_bytes())
    owner = doc["passive_tissue_parent_bindings"]
    require(owner["child_payload_sha256"] == proof["input"], "Unmatched child receipt")
    owner["child_payload_sha256"] = proof["output"]
    out_receipt = output / receipt.name
    out_receipt.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    require(digest(out_receipt) == proof["amended_receipt"], "Changed unrelated anatomical bindings")
    for name, expected in proof["cardiac_fields"].items():
        require(Path(name).name == name, "Unscoped cardiac field")
        src, dst = receipt.parent / name, output / name
        require(digest(src) == expected, "Unmatched cardiac deformation coefficients")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        require(digest(dst) == expected, "Incomplete cardiac sidecar")
    return target


def rebuild(parent, donor, atlas, proposal, proof, output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".teres-") as temporary:
        stage = Path(temporary) / "scene"
        target = build_staged(parent, donor, atlas, proposal, proof, stage)
        stage.rename(output)
    return output / target.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "donor", "atlas", "proposal", "proof", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(rebuild(**vars(args)))
