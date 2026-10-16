"""Minimal inferred-hand skin topology repair over exact retained native Float32 points."""

from pathlib import Path
import sys, json, hashlib, struct, time, traceback
import numpy as np

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "src"))
from numilab_human import (
    resting_anatomy_verify as v,
    skin_source_payload_preflight as sp,
    common_atlas_skin_geometry_registration as gr,
    gate_preparation_boundary as gp,
)
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--evidence-root",
    type=Path,
    required=True,
    help="Retained full-anatomy-delivery-audit-1378 directory",
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
D = args.evidence_root.resolve()
I = D / "skin864-native003-self-witnesses-887b"
N = I / "actual-native-skin.npz"
F = D / "combined-skin-native003-unified-self-verification-886/strict-reference.json"
B = D / "skin729-existing-full86-source-export-864"
S = B / "bodyparts3d-myosim-skinned-shell.nhskin"
REG = (
    D
    / "current-manifest-recovery-004/bilateral-regional-b42-current-source-compatibility-001/candidate-current-source.json"
)
O = args.output.resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


assert not O.exists()
# This recipe applies only to the authenticated retained source/native witness pair.
assert sha(S) == "9504692bb42e347a79c661b6311160a559d5fd040c0f944c61bcfd5d21cedc07"
assert sha(N) == "f990de5b133e42825e02cd7120638e060922329fcc9c882d9f9e816f16abbb83"
assert sha(F) == "30f791d3f2b876701e10bc54918ca1eed8735957e8e6d41328a0ca62c41d24f2"
pre = gp.freeze_preparation(
    O.with_suffix(".preparation.json"),
    dict(
        status="PASS_PINNED_INFERRED_HAND_SKIN_TOPOLOGY_PREPARATION",
        predicate_phase_started=False,
        scope="Two oriented local edge flips plus one link-condition edge collapse of a 75nm native inferred hand edge. Preserve every source allocation, XYZ, full86 weight and binding. Retire one inferred source slot from active triangle references only; preserve original slot bytes and ancestry. Recompute normals through existing owner locally to hands. Every other source vertex record and working back geometry byteexact. No force/mass/physiology change; no anatomical/native admission.",
    ),
    runner_argv=[sys.executable, *sys.argv],
    input_pins={
        str(p): sha(p)
        for p in (
            Path(__file__),
            N,
            I / "report.json",
            F,
            S,
            B / "refinement-proof.json",
            REG,
            *[Path(m.__file__) for m in (v, sp, gr, gp)],
        )
    },
)
gp.verify_frozen_preparation(pre)
O.mkdir()
start = time.monotonic()
try:
    failure = json.loads(F.read_text())
    assert (
        failure["failed_stage"] == "self"
        and len(failure["rows"][0]["triangle_pairs"]) == 9
    )
    with np.load(N, allow_pickle=False) as z:
        world = z["capture_positions_m"]
        faces = z["capture_faces"]
        gids = z["capture_global_vertex_ids"]
        slots = z["raw_source_slots"]
        normals = z["capture_normals"]
    raw = S.read_bytes()
    base = sp.decode_payload(raw)
    sf = base["indices"].reshape(-1, 3).copy()
    assert np.array_equal(slots[faces], sf)
    oldsf = sf.copy()
    byraw = np.full(base["vertex_count"], -1, dtype=np.int64)
    byraw[slots] = np.arange(len(slots))
    ancestry = [[i] for i in range(len(sf))]
    operations = []
    touched = set()
    for u, k, want_v, w in ((56896, 56931, 56943, 56982), (56856, 56868, 56875, 56888)):
        ids = np.flatnonzero(np.sum(np.isin(sf, (u, want_v)), axis=1) == 2)
        assert len(ids) == 2
        sliver = [int(i) for i in ids if set(sf[i]) == {u, k, want_v}]
        outer = [int(i) for i in ids if set(sf[i]) == {u, want_v, w}]
        assert len(sliver) == len(outer) == 1
        a, b = sliver[0], outer[0]

        def cyclic(row, expected):
            return any(tuple(np.roll(row, -j)) == tuple(expected) for j in range(3))

        if cyclic(sf[b], (u, want_v, w)):
            assert cyclic(sf[a], (u, k, want_v))
            sf[a] = [u, k, w]
            sf[b] = [k, want_v, w]
        else:
            assert cyclic(sf[b], (u, w, want_v)) and cyclic(sf[a], (u, want_v, k))
            sf[a] = [u, w, k]
            sf[b] = [k, w, want_v]
        ancestry[a] = ancestry[b] = sorted((a, b))
        touched.update((u, k, want_v, w))
        operations.append(
            dict(
                operation="oriented_edge_flip",
                parent_faces=[a, b],
                old_edge=[u, want_v],
                new_edge=[k, w],
                source_vertex_xyz_and_weights_unchanged=True,
            )
        )
    keep, drop = 61619, 61620
    incident_keep = sf[np.any(sf == keep, axis=1)]
    incident_drop = sf[np.any(sf == drop, axis=1)]
    neighbors_keep = set(incident_keep.ravel()) - {keep}
    neighbors_drop = set(incident_drop.ravel()) - {drop}
    edgefaces = np.flatnonzero(np.sum(np.isin(sf, (keep, drop)), axis=1) == 2)
    assert len(edgefaces) == 2 and set(edgefaces.tolist()) == {104074, 104475}
    opposite = set(sf[edgefaces].ravel()) - {keep, drop}
    assert (neighbors_keep & neighbors_drop) == opposite, "edge link condition failed"
    actual_edge = float(
        np.linalg.norm(
            world[byraw[keep]].astype(float) - world[byraw[drop]].astype(float)
        )
    )
    assert actual_edge < 2e-7
    touched.update(incident_keep.ravel().tolist())
    touched.update(incident_drop.ravel().tolist())
    changed_faces = np.flatnonzero(np.any(sf == drop, axis=1))
    sf[sf == drop] = keep
    live = np.ones(len(sf), dtype=bool)
    live[edgefaces] = False
    assert np.all(np.sum(sf[live][:, 0, None] == sf[live], axis=1) == 1)
    operations.append(
        dict(
            operation="link_valid_inferred_edge_collapse",
            keep_source_slot=keep,
            retired_active_source_slot=drop,
            native_edge_length_m=actual_edge,
            retired_parent_faces=edgefaces.tolist(),
            changed_parent_faces=changed_faces.tolist(),
            all_old_source_allocations_and_values_retained=True,
        )
    )
    new_to_old = np.flatnonzero(live)
    newfaces = sf[live]
    local = byraw[newfaces]
    assert np.all(local >= 0)
    assert all(
        any(face in touched_faces for face in pair)
        for pair in failure["rows"][0]["triangle_pairs"]
        for touched_faces in [
            set(np.flatnonzero(np.any(oldsf != sf, axis=1)).tolist())
            | set(edgefaces.tolist())
        ]
    ), "old witness not covered by repair"
    # Preparation: topological repairs are confined to actual hands, with no source position edit.
    used_touch = np.array(sorted(touched))
    assert np.all(byraw[used_touch] >= 0)
    hand = world[byraw[used_touch]]
    assert np.all(hand[:, 1] < -0.9) and np.all(np.abs(hand[:, 0]) > 0.15)
    surface = v.prepare_surface(
        (51007, 1), world, local.astype("<u4"), normals=normals, source_vertex_ids=gids
    )
    before_topology = v.analyze_topology(world.tolist(), faces.tolist())
    after_topology = surface.raw_topology
    assert (
        len(before_topology["boundary_edges"])
        == len(after_topology["boundary_edges"])
        == 115
    )
    for key in (
        "nonmanifold_edges",
        "orientation_defect_edges",
        "degenerate_face_ids",
        "repeated_vertex_face_ids",
        "duplicate_face_ids",
        "vertex_manifold_defect_ids",
        "boundary_branch_vertex_ids",
    ):
        assert not after_topology[key], key
    audit = surface.self_audit()
    dump(O / "actual-f32-self.json", audit)
    assert audit["count"] == 0, "first failed geometry gate: actual Float32 self"
    np.savez_compressed(
        O / "candidate.npz",
        capture_positions_m=world,
        capture_faces=local.astype("<u4"),
        capture_global_vertex_ids=gids,
        raw_source_slots=slots,
        capture_normals=normals,
    )
    # Encode existing NHSKIN ABI5; only index_count/topology and local normal bytes may differ.
    head = bytearray(raw[:60])
    struct.pack_into("<I", head, 20, int(newfaces.size))
    be = 60 + 36 * base["binding_count"]
    intermediate = (
        bytes(head)
        + raw[60:be]
        + base["vertices_u"].tobytes()
        + newfaces.astype("<u4").tobytes()
        + base["full_weights"].tobytes()
    )
    rebuilt, normal_proof = gr.compose_disjoint_skin_position_corrections(
        intermediate,
        [intermediate],
        global_source_matrix=json.loads(REG.read_text())["coordinate_system"][
            "global_source_mm_to_myosim_world_m"
        ],
    )
    affected = used_touch.copy()
    for _ in range(3):
        affected = np.unique(newfaces[np.any(np.isin(newfaces, affected), axis=1)])
    # Normal reconstruction is allowed only inside the hand supports, and all untouched vertex bytes are restored.
    assert np.all(byraw[affected] >= 0)
    hand_support = world[byraw[affected]]
    assert np.all(hand_support[:, 1] < -0.88) and np.all(
        np.abs(hand_support[:, 0]) > 0.14
    )
    final = bytearray(rebuilt)
    untouched = np.setdiff1d(np.arange(base["vertex_count"]), affected)
    for i in untouched:
        final[be + int(i) * 56 : be + (int(i) + 1) * 56] = raw[
            be + int(i) * 56 : be + (int(i) + 1) * 56
        ]
    out = sp.decode_payload(bytes(final))
    assert np.array_equal(out["vertices_u"][:, :3], base["vertices_u"][:, :3])
    assert np.array_equal(
        out["vertices_u"][:, 6:], base["vertices_u"][:, 6:]
    ) and np.array_equal(out["full_weights"], base["full_weights"])
    assert final[60:be] == raw[60:be] and np.array_equal(
        out["vertices_u"][untouched], base["vertices_u"][untouched]
    )
    assert np.isfinite(out["vertices_f"][:, :6]).all()
    path = O / S.name
    path.write_bytes(final)
    lineage = [
        dict(
            output_face=i,
            parent_current_face_ordinals=ancestry[int(old)],
            operation=(
                "unchanged"
                if len(ancestry[int(old)]) == 1
                and np.array_equal(newfaces[i], oldsf[old])
                else "documented_local_hand_topology_repair"
            ),
        )
        for i, old in enumerate(new_to_old)
        if len(ancestry[int(old)]) > 1 or not np.array_equal(newfaces[i], oldsf[old])
    ]
    proof = dict(
        parent_skin_refinement_proof=str(B / "refinement-proof.json"),
        parent_skin_refinement_sha256=sha(B / "refinement-proof.json"),
        parent_skin_payload_path=str(S),
        parent_skin_payload_sha256=sha(S),
        operations=operations,
        changed_face_lineage=lineage,
        output_face_to_parent_current_face=new_to_old.tolist(),
        normal_owner_proof=normal_proof,
        retired_source_slots=[drop],
        all_source_xyz_allocations_bindings_full86_weights_exact=True,
        normal_affected_source_slots=affected.tolist(),
        all_other_source_vertex_records_exact=True,
        working_back_records_exact_outside_explicit_hand_support=True,
        anatomical_admission=False,
        native_admission=False,
    )
    dump(O / "refinement-proof.json", proof)
    result = dict(
        status="PASS_OFFLINE_EXISTING_SKIN_SOURCE_EXPORT_ONLY",
        source_path=str(path),
        source_sha256=sha(path),
        parent_source_sha256=sha(S),
        vertex_count=base["vertex_count"],
        face_count=len(newfaces),
        births=0,
        corrected_original_slots=0,
        retired_active_inferred_source_slots=[drop],
        protected_back_source_vertex_records_exact=True,
        original_binding_and_full86material_rows_exact=True,
        all_source_positions_byteexact=True,
        physical_mass_contact_and_physiology_owners_unchanged=True,
        exact_native_emission=False,
        anatomical_admission=False,
        native_admission=False,
        reference_assembly="Registered mixed-source reference adult; documented inferred local hand topology repair, not measured-person anatomy.",
        elapsed_seconds=time.monotonic() - start,
        next_change="Rebind exact existing scene/receipt skin identities without changing mass or contact; fresh native capture, allnear and nine retained GPU poses.",
    )
    dump(O / "source-preparation-report.json", result)
    gp.verify_frozen_preparation(pre)
    print(json.dumps(result), flush=True)
except BaseException as e:
    dump(
        O / "failure.json",
        dict(
            error=repr(e),
            traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - start,
        ),
    )
    raise
