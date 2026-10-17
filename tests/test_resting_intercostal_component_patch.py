import copy
import numpy as np
import pytest
from numilab_human.resting_intercostal_mesh_arrangement import (
    compose_local_reconstruction_component_patch as compose,
)


def scene():
    p = np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                    [3,0,0],[4,0,0],[3,1,0],[3,0,1]], dtype="<f4")
    tetra = np.asarray([[0,2,1],[0,1,3],[1,2,3],[2,0,3]], dtype="<u4")
    parent = dict(positions_source_local_f32=p,
                  positions_world_f32=p + np.asarray([0,0,10], dtype="<f4"),
                  faces=np.concatenate([tetra, tetra + 4]))
    patch = {k: v[:4].copy() for k, v in parent.items()}
    patch["faces"] = tetra.copy()
    patch["positions_source_local_f32"][0] = [.125,.125,.125]
    patch["positions_world_f32"][0] = [.125,.125,10.125]
    return parent, patch


def call(parent, patch, **kw):
    return compose(parent, patch, stable_id=313,
                   replaced_parent_face_ids=kw.get("selected", [0,1,2,3]),
                   patch_parent_face_ids=kw.get("parents", [[0],[1],[2],[3]]))


def test_complete_body_component_survives_local_patch():
    parent, patch = scene()
    original = copy.deepcopy(parent)
    out, report = call(parent, patch, parents=[[0,1],[1],[2],[3,0]])
    assert len(out["faces"]) == 8  # A local patch is never the complete row.
    assert out["faces"][:4].tobytes() == parent["faces"][4:].tobytes()
    for key in ("positions_source_local_f32", "positions_world_f32"):
        assert out[key][:8].tobytes() == original[key].tobytes()
        assert out[key][out["faces"][4:]].tobytes() == patch[key][patch["faces"]].tobytes()
        assert parent[key].tobytes() == original[key].tobytes()
    assert out["face_parent_offsets_u64"].tolist() == [0,1,2,3,4,6,7,8,10]
    assert out["face_parent_ids_u32"].tolist() == [4,5,6,7,0,1,1,2,0,3]
    assert report["retained_parent_face_count"] == 4
    assert not any(report[k] for k in ("anatomical_admission", "native_admission",
                                      "attachment_map_admission"))


def test_existing_unused_source_allocation_is_retained():
    parent, patch = scene()
    for key in ("positions_source_local_f32", "positions_world_f32"):
        parent[key] = np.concatenate([parent[key], parent[key][7:8]])
    out, _ = call(parent, patch)
    assert out["positions_source_local_f32"][:9].tobytes() == parent["positions_source_local_f32"].tobytes()
    assert len(out["positions_source_local_f32"]) == 10


@pytest.mark.parametrize("selected,parents", [
    ([0,0,2,3], [[0],[1],[2],[3]]),
    ([0,1,2,3], [[4],[1],[2],[3]]),
    ([0,1,2,3], [[0],[],[2],[3]]),
    ([0,1,2,3], [[0],[1],[2]]),
    ([True,1,2,3], [[0],[1],[2],[3]]),
])
def test_rejects_incomplete_or_out_of_region_lineage(selected, parents):
    with pytest.raises(ValueError):
        call(*scene(), selected=selected, parents=parents)


@pytest.mark.parametrize("bad", ["float64", "nonfinite", "index", "degenerate", "world_alias"])
def test_rejects_invalid_emission_before_composition(bad):
    parent, patch = scene()
    if bad == "float64":
        patch["positions_source_local_f32"] = patch["positions_source_local_f32"].astype("<f8")
    elif bad == "nonfinite":
        patch["positions_source_local_f32"][0,0] = np.nan
    elif bad == "index":
        patch["faces"][0,0] = 1000
    elif bad == "degenerate":
        patch["faces"][0,0] = patch["faces"][0,1]
    else:
        patch["positions_world_f32"][1,0] += .25
    with pytest.raises(ValueError):
        call(parent, patch)
