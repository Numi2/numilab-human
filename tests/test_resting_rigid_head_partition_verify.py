"""The shared verifier binds rigid head interfaces to the accepted NHA owner."""
import hashlib
import copy
import numpy as np
import pytest
from numilab_human import resting_shared_partition_verify as shared
from numilab_human.resting_anatomy_interface_patch import HEADER, RECORD
from tests.test_resting_shared_partition_verify import scene


def head_scene(tmp_path, ids=(12, 323)):
    _, _, muscle = scene(tmp_path)
    captures, vertices, faces, records = {}, [], [], []
    for sid, old in zip(ids, (1, 2)):
        x, f, n, gids = muscle[(51005, old)]
        first = sum(len(v) for v in vertices)
        index = sum(v.size for v in faces)
        records.append(RECORD.pack(23, first, len(x), index, f.size, sid, 3 if sid == 12 else 1, 0))
        vertices.append(np.c_[x, n].astype("<f4"))
        faces.append((f+first).astype("<u4"))
        captures[(51012 if sid == 12 else 51010, sid)] = (x, f, n, np.arange(len(x), dtype="<u4")+first+10000)
    raw = (HEADER.pack(b"NHANAT1\0", 5, 2, 8, 24, 17, b"x"*32)
           + b"".join(records)+b"".join(v.tobytes() for v in vertices)
           + b"".join(f.tobytes() for f in faces))
    source = tmp_path/"head.nhanatomy";source.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    context = {"capture_receipt": {"common_field_source_anatomy_payload_sha256": digest,
                "accepted_body_poses": [{"body_index":23,"position_m":[0,0,0],
                                         "quaternion_xyzw":[0,0,0,1]}]},
               "capture_receipt_sha256":"a"*64}
    return source, digest, captures, context


@pytest.mark.parametrize("ids", [(12,323), (12,343), (323,343), (323,345), (343,345)])
def test_exact_rigid_head_certificate_uses_existing_partition_predicate(tmp_path, ids):
    source, digest, captures, context = head_scene(tmp_path, ids)
    first, second = ((51012 if sid == 12 else 51010, sid) for sid in ids)
    proof = shared.certify_shared_muscle_partition(
        captures, first, second, source, digest, capture_context=context)
    assert proof["shared_interface_face_count"] == 1
    assert proof["exact_native_partition_certificate"]["interiors_disjoint"]
    assert not proof["anatomical_admission"]


def test_cold_head_preparation_binds_payload_pose_and_facets_without_predicates(tmp_path, monkeypatch):
    source, digest, captures, context = head_scene(tmp_path)
    monkeypatch.setattr(shared.cert, "audit_shared_interface_partition",
                        lambda *a, **k: pytest.fail("cold preparation invoked exact predicates"))
    def check(c=captures, ctx=context, first=(51012,12), second=(51010,323)):
        return shared.preflight_shared_muscle_partition(
            c, first, second, source, digest, capture_context=ctx)
    assert not check()["exact_predicates_executed"]
    bad = copy.deepcopy(context)
    bad["capture_receipt"]["common_field_source_anatomy_payload_sha256"] = "b"*64
    with pytest.raises(Exception, match="source differs from accepted capture"):check(ctx=bad)
    bad = copy.deepcopy(context);bad["capture_receipt"]["accepted_body_poses"] *= 2
    with pytest.raises(Exception, match="unique accepted head pose"):check(ctx=bad)
    with pytest.raises(Exception, match="typed interface is unsupported"):
        check(first=(51010,343),second=(51010,374))
    captures[(51010,323)][0][0,0] = np.float32(1e-8)
    with pytest.raises(Exception, match="different Float32 positions"):check()


def test_head_partition_rejects_missing_context_and_reordered_source_facets(tmp_path):
    source, digest, captures, context = head_scene(tmp_path)
    with pytest.raises(Exception, match="accepted capture context"):
        shared.preflight_shared_muscle_partition(captures,(51012,12),(51010,323),source,digest)
    captures[(51010,323)][1][0,[1,2]] = captures[(51010,323)][1][0,[2,1]]
    with pytest.raises(Exception):
        shared.preflight_shared_muscle_partition(
            captures,(51012,12),(51010,323),source,digest,capture_context=context)
