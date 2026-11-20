"""Revalidate the retained P1 mapping at every displaced Float32 node."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root.parent / "resting-clearance-and-atrium-016"))
from screen_paired_mass_routes import COMPILER, sha
from export_repaired_tetgen import array_hash

prior = root / "complete-fine-p1-live-parent"
prior_state = prior / "current-paired-volume-state.json"
state = json.loads(prior_state.read_text())
floor = root / "constitutive-witness-repaired-tetgen"
mesh_receipt = json.loads((floor / "receipt.json").read_text())
assert sha(floor / "mesh.node") == mesh_receipt["mesh_node_sha256"]
points = np.loadtxt(floor / "mesh.node", skiprows=1, usecols=(1,2,3), dtype=np.float32)
moved = np.asarray(json.loads((root / "constitutive-witness-surface-delta.json").read_text())["moved_nodes"])
old_points = np.loadtxt(root / "absolute-volume-repaired-tetgen/mesh.node",
                        skiprows=1, usecols=(1,2,3), dtype=np.float32)
assert array_hash(old_points, "<f4") == state["geometry_sha256"]
assert np.array_equal(np.flatnonzero(np.any(points != old_points,axis=1)),moved)
entry = Path(json.loads((COMPILER / "p1-Skin.command.json").read_text())["command"][1])
owner_sha = json.loads((root.parent / "resting-clearance-and-atrium-013/fine-local-current-p1.json").read_text())["owner_sha256"]
assert sha(entry) == owner_sha
spec = importlib.util.spec_from_file_location("pinned_role_p1",entry)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
owner = api._load_230(api.OWNER230)
# The native live-parent packet already certified these preferences from
# their historical projection frame. Recompute the displaced points in
# that same frame, then extend the live bound by the actual move.
parent_pins = state["historical_preference_source330"]
parent_mass = Path(parent_pins["source330_mass"]["path"])
parent_nodes = Path(parent_pins["source330_nodes"]["path"])
assert sha(parent_mass) == parent_pins["source330_mass"]["sha256"]
assert sha(parent_nodes) == parent_pins["source330_nodes"]["sha256"]
parent = owner.native.read_parent(parent_mass)
parent["points"] = np.loadtxt(parent_nodes,delimiter=",",skiprows=1,usecols=(2,3,4))
index = owner.native.build_index(parent)
material = owner.build_material_indices(parent)
out = root / "refit-motion-qualification/p1"
out.mkdir(exist_ok=True)
new_state = out / prior_state.name
state["geometry_sha256"] = array_hash(points,"<f4")
state["prior_geometry_state_sha256"] = sha(prior_state)
state["geometry_receipt"] = {"path":str(floor / "receipt.json"),"sha256":sha(floor / "receipt.json")}
maximum_move = float(np.linalg.norm(points[moved].astype(float) - old_points[moved], axis=1).max())
for role in ("skin", "fat"):
    key = f"p1_{role}_current_distance_upper_bound_m"
    state[key] += maximum_move
    assert state[key] <= state["p1_current_distance_limit_m"]
new_state.write_text(json.dumps(state,indent=2)+"\n")
state_sha = sha(new_state)
for role in ("skin","fat"):
    info = json.loads((prior / f"{role}-p1-receipt.json").read_text())
    groups,lookup = api._source_groups(parent,role.title())
    geometry = Path(info["role_geometry_path"])
    if not geometry.exists():
        geometry = out.parent / geometry.name
        if not geometry.exists():
            cells = np.loadtxt(floor / "mesh.ele", skiprows=1, usecols=(1,2,3,4,5), dtype=np.int32)
            rows = np.flatnonzero(cells[:,4] == api.ROLE_MATERIAL[role.title()])
            nodes, local_cells = np.unique(cells[rows,:4]-1, return_inverse=True)
            np.savez_compressed(geometry, points_world_m_f32=old_points[nodes],
                                tetrahedra=local_cells.reshape(-1,4).astype("<u4"),
                                combined_node_rows=nodes, combined_child_rows=rows)
    assert sha(geometry) == info["role_geometry_sha256"]
    with np.load(geometry) as packet:
        geom = dict(packet)
    role_nodes = geom["combined_node_rows"]
    local = np.flatnonzero(np.isin(role_nodes,moved))
    assert np.array_equal(geom["points_world_m_f32"],old_points[role_nodes])
    projected = api.build_node_support(points[role_nodes[local]],role.title(),parent,index,material,groups,lookup,owner)
    old_base_digest = info["p1_sha256"]
    base_path = Path(info["p1_path"])
    assert sha(base_path) == old_base_digest
    with np.load(base_path) as old:
        offsets = old["node_group_offsets"]
        group = old["node_group_ids"]
        source_starts = old["node_group_parent_tet_offsets"]
        source_parents = old["node_group_parent_tet_rows"]
        for j,row in enumerate(local):
            mapping = {group[s]:set(source_parents[source_starts[s]:source_starts[s+1]])
                       for s in range(offsets[row],offsets[row+1])}
            if mapping != projected["maps"][j]:
                raise ValueError(f"{role}: geometric projection changed exact ancestry at node {role_nodes[row]}")
    geom["points_world_m_f32"] = points[role_nodes]
    geom_path = out / geometry.name
    np.savez_compressed(geom_path,**geom)
    geom_sha = sha(geom_path)
    department = out / role.title()
    department.mkdir(exist_ok=True)
    replacement_base = None
    for path_key,hash_key in (("p1_path","p1_sha256"),("p1_one_hop_path","p1_one_hop_sha256")):
        source_path = Path(info[path_key])
        assert sha(source_path) == info[hash_key]
        with np.load(source_path) as old:
            data = dict(old)
        assert str(data["current_state_sha256"]) == sha(prior_state)
        assert str(data["geometry_sha256"]) == info["role_geometry_sha256"]
        assert np.array_equal(data["points_world_m_f32"],old_points[role_nodes])
        data["points_world_m_f32"] = geom["points_world_m_f32"]
        data["node_parent_registration_distance_m"][local] = projected["distance"]
        data["current_state_path"] = np.asarray(str(new_state))
        data["current_state_sha256"] = np.asarray(state_sha)
        data["geometry_sha256"] = data["volume_sha256"] = np.asarray(geom_sha)
        if replacement_base is not None:
            assert str(data["admissibility_source_support_sha256"]) == old_base_digest
            data["admissibility_source_support_sha256"] = np.asarray(replacement_base)
        destination = department / source_path.name
        np.savez_compressed(destination,**data)
        new_hash = sha(destination)
        if replacement_base is None:
            replacement_base = new_hash
        info[path_key] = str(destination)
        info[hash_key] = new_hash
    info.update(state_path=str(new_state),state_sha256=state_sha,
                role_geometry_path=str(geom_path),role_geometry_sha256=geom_sha,
                changed_nodes_exactly_reprojected=role_nodes[local].tolist(),
                parent_maps_identical=True)
    (out / f"{role}-p1-receipt.json").write_text(json.dumps(info,indent=2)+"\n")
    print(role, "exact P1 ancestry and region sets unchanged at", role_nodes[local].tolist(),flush=True)
