"""Regression checks for the existing NHBONES local partition encoder."""
import numpy as np
import pytest
from numilab_human import composed_bone_body_frame as bone, model


def fixture():
    xyz = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype="<f4")
    attrs = np.column_stack((xyz, np.tile([0,0,1], (4,1)))).astype("<f4")
    faces = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]], dtype="<u4")
    rows, points, triangles = [], [], []
    for sid in range(1,186):
        pose = [.25,.125,.5,0,0,0,1,1] if sid == 50 else [0,0,0,0,0,0,1,1]
        rows.append((20,4*(sid-1),4,12*(sid-1),12,sid,*pose,sid-1))
        points.append(attrs.tobytes())
        triangles.append((faces+4*(sid-1)).astype("<u4").tobytes())
    raw = (bone._BONE_HEADER.pack(b"NHBONES1",3,185,740,2220,123,b"x"*32)
           + b"".join(bone._BONE_RECORD.pack(*r) for r in rows)
           + b"".join(points)+b"".join(triangles))
    old = np.asarray([bone._row_body_local(tuple(a),tuple(rows[49][6:14]))
                      for a in attrs], dtype="<f4")
    midpoint = old[:3].astype(float).mean(axis=0).astype("<f4")
    new = np.vstack((old, midpoint))
    new_faces = np.array([[0,2,4],[2,1,4],[1,0,4],*faces[1:]], dtype="<u4")
    return raw,new,new_faces


def test_partition_encoder_retains_all_other_geometry_and_owners():
    raw,attrs,faces = fixture()
    output,proof = bone.replace_bone_source_partition(
        raw, stable_id=50, vertices6=attrs, faces=faces)
    h,rows,vo,io = bone._decode_bones(raw)
    nh,nrows,nv,ni = bone._decode_bones(output)
    assert nh[3] == h[3]+1 and nh[4] == h[4]+6
    for old,new in zip(rows,nrows,strict=True):
        assert old[0] == new[0] and old[5] == new[5] and old[14] == new[14]
        if old[5] == 50:
            assert new[6:14] == (0.,0.,0.,0.,0.,0.,1.,1.)
            assert output[nv+new[1]*24:nv+(new[1]+new[2])*24] == attrs.tobytes()
        else:
            assert old[2] == new[2] and old[4:] == new[4:]
            assert raw[vo+old[1]*24:vo+(old[1]+old[2])*24] == output[nv+new[1]*24:nv+(new[1]+new[2])*24]
            a=np.frombuffer(raw,"<u4",count=old[4],offset=io+old[3]*4)-old[1]
            b=np.frombuffer(output,"<u4",count=new[4],offset=ni+new[3]*4)-new[1]
            assert np.array_equal(a,b)
    assert proof["unchanged_other_bone_geometries"] == 184
    assert proof["physical_mass_or_state_changed"] is False
    assert proof["native_admission"] is False


@pytest.mark.parametrize("mutation", ["position","normal","index","degenerate","wrong_id"])
def test_partition_encoder_rejects_source_or_topology_drift(mutation):
    raw,attrs,faces = fixture()
    sid=50
    if mutation == "position": attrs[0,0] += .001
    elif mutation == "normal": attrs[0,3] += .001
    elif mutation == "index": faces[0,0] = len(attrs)
    elif mutation == "degenerate": faces[0,1] = faces[0,0]
    else: sid=49
    with pytest.raises(model.ImportError):
        bone.replace_bone_source_partition(raw,stable_id=sid,vertices6=attrs,faces=faces)

