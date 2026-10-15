from pathlib import Path
import hashlib
import numpy as np
import pytest
from tests.test_passive_reference_binding_extension import fixture
from numilab_human import passive_attachment_composition as pac
from numilab_human.resting_shared_partition_verify import certify_shared_muscle_partition


def scene(tmp_path):
    source = fixture(tmp_path)
    data = pac._read_nhtiss4(source)
    raw = bytearray(data["raw"])
    row = data["records"][1]
    a = pac._nhtiss_row_arrays(data, row)
    offset = data["vertex_start"]+int(row[2])*56
    vertices = np.ndarray((4, 3), "<f4", buffer=raw, offset=offset, strides=(56, 4))
    vertices[3, 2] = -1
    # Reflecting one tetrahedron reverses every facet orientation.
    faces = np.ndarray((4, 3), "<u4", buffer=raw,
                       offset=data["index_start"]+int(row[4])*4)
    faces[:, [1, 2]] = faces[:, [2, 1]]
    source.write_bytes(raw)
    data = pac._read_nhtiss4(source)
    captures = {}
    for row in data["records"][:2]:
        a = pac._nhtiss_row_arrays(data, row)
        gid = np.arange(a["vertex_count"], dtype="<u4")+a["first_vertex"]+10000
        captures[(51005, int(row[6]))] = (
            a["positions"].copy(), a["faces"].copy(), a["normals"].copy(), gid)
    return source, hashlib.sha256(raw).hexdigest(), captures


def test_native_partition_requires_exact_source_and_shared_material(tmp_path):
    source, digest, captures = scene(tmp_path)
    proof = certify_shared_muscle_partition(captures, (51005,1), (51005,2), source, digest)
    assert proof["shared_interface_face_count"] == 1
    assert proof["exact_native_partition_certificate"]["interiors_disjoint"]
    assert proof["exact_native_partition_certificate"]["shared_interface_opposite_chain"]
    assert not proof["physical_state_advanced"]


def test_rejects_native_interface_coordinate_drift(tmp_path):
    source, digest, captures = scene(tmp_path)
    captures[(51005,2)][0][0, 0] = np.float32(1e-8)
    with pytest.raises(Exception, match="different native Float32 positions"):
        certify_shared_muscle_partition(captures, (51005,1), (51005,2), source, digest)


def test_rejects_coincident_geometry_with_different_material_owner(tmp_path):
    source, digest, captures = scene(tmp_path)
    data = pac._read_nhtiss4(source)
    raw = bytearray(data["raw"])
    row = data["records"][1]
    slots = np.ndarray((4, 4), "<u4", buffer=raw,
                       offset=data["vertex_start"]+int(row[2])*56+24, strides=(56, 4))
    slots[:, 0] = 1
    source.write_bytes(raw)
    with pytest.raises(Exception, match="no reciprocal source/material shared interface"):
        certify_shared_muscle_partition(captures, (51005,1), (51005,2), source,
                                       hashlib.sha256(raw).hexdigest())


def test_rejects_stale_source_hash_and_ordered_facet_map(tmp_path):
    source, digest, captures = scene(tmp_path)
    with pytest.raises(Exception, match="source hash differs"):
        certify_shared_muscle_partition(captures, (51005,1), (51005,2), source, "0"*64)
    captures[(51005,2)][1][0, [1,2]] = captures[(51005,2)][1][0, [2,1]]
    with pytest.raises(Exception):
        certify_shared_muscle_partition(captures, (51005,1), (51005,2), source, digest)
