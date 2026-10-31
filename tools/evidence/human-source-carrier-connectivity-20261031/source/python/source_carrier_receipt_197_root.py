"""Truthful receipt-mode checks for source-only carrier allocation verification."""
from __future__ import annotations

FLOW_NAMES = {
    "transport-child-mass.csv", "transport-flow.csv", "source-carrier-flow.csv",
    "transport-child-source-donor.csv", "transport-source-parent.csv",
    "transport-source-parent-donor.csv",
}
LP_SUFFIX = "SOURCE_CARRIER_FEASIBLE_ALLOCATION"
LEGACY_SUFFIX = "SOURCE_CARRIER_SHARED_CAPACITY_ALLOCATION"

def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid SHA-256 field: {field}")
    return value


class CanonicalDigest:
    """Mirror the existing C++ SHA256Builder big-endian scalar encoding."""
    def __init__(self):
        import hashlib
        self._hash = hashlib.sha256()
    def bytes(self, value: bytes) -> None:
        self._hash.update(value)
    def string(self, value: str) -> None:
        import struct
        data = value.encode("utf-8")
        self.u64(len(data))
        self.bytes(data)
    def u32(self, value: int) -> None:
        import struct
        if not 0 <= int(value) <= 0xffffffff:
            raise ValueError("u32 value outside range")
        self.bytes(struct.pack(">I", int(value)))
    def u64(self, value: int) -> None:
        import struct
        if not 0 <= int(value) <= 0xffffffffffffffff:
            raise ValueError("u64 value outside range")
        self.bytes(struct.pack(">Q", int(value)))
    def real(self, value: float) -> None:
        import struct
        self.bytes(struct.pack(">d", float(value)))
    def finish(self) -> str:
        return self._hash.hexdigest()

def source_carrier_packet_digest(base_packet_sha, held_manifest_sha, held_flow_sha,
                                 lp_proposal_sha, lp_manifest_sha, offsets,
                                 global_source_rows, weights, barycentric,
                                 witness_child_rows, witness_distances):
    """Recompute the finalizer's composite packet digest from its canonical arrays."""
    h = CanonicalDigest()
    h.string("numi.human.source-carrier-held-plus-lp-packet.v1")
    for value in (base_packet_sha, held_manifest_sha, held_flow_sha,
                  lp_proposal_sha, lp_manifest_sha):
        _sha(value, "composite packet provenance")
        h.string(value)
    h.u64(len(offsets))
    for x in offsets:h.u32(x)
    h.u64(len(global_source_rows))
    for x in global_source_rows:h.u32(x)
    h.u64(len(weights))
    for x in weights:h.real(x)
    for row in barycentric:
        for x in row:h.real(x)
    for x in witness_child_rows:h.u32(x)
    for x in witness_distances:h.real(x)
    return h.finish()

def source_carrier_array_digest(packet_sha, source_tet_table_sha, source_node_table_sha,
                                offsets, global_source_rows, source_tet_node_rows,
                                parent_tet_world_vertices, weights, barycentric,
                                witness_child_rows, witness_distances, parent_count,
                                parent_mass_packet_sha, parent_source_sha,
                                child_mesh_sha, region_label_sha, coordinate_frame_sha):
    """Recompute the C++ source-carrier array digest for the finalized packet."""
    h=CanonicalDigest()
    h.string("numi.human.source-tetra-centroid-carrier-quadrature.arrays.v1")
    for value in (packet_sha,source_tet_table_sha,source_node_table_sha):
        _sha(value,"carrier array provenance")
        h.string(value)
    h.u64(len(offsets))
    for x in offsets:h.u32(x)
    h.u64(len(global_source_rows))
    for x in global_source_rows:h.u32(x)
    for tet in source_tet_node_rows:
        for x in tet:h.u32(x)
    for tet in parent_tet_world_vertices:
        for point in tet:
            for x in point:h.real(x)
    h.u64(len(weights))
    for x in weights:h.real(x)
    for row in barycentric:
        for x in row:h.real(x)
    h.u64(len(witness_child_rows))
    for x in witness_child_rows:h.u32(x)
    for x in witness_distances:h.real(x)
    h.u64(parent_count)
    for value in (parent_mass_packet_sha,parent_source_sha,child_mesh_sha,
                  region_label_sha,coordinate_frame_sha):
        _sha(value,"carrier array input")
        h.string(value)
    return h.finish()

def validate_receipt_mode(receipt: dict, role: str) -> str:
    if role not in {"Skin", "Fat"}:
        raise ValueError("role must be Skin or Fat")
    role_upper = role.upper()
    status = receipt.get("status")
    if status == f"PASS_SOURCE_ONLY_{role_upper}_{LP_SUFFIX}":
        required = {
            "complete_allocation": True,
            "physical_admission": False,
            "native_steps": 0,
            "recorded_flow_feasibility": "PASS_SOLVER_FREE",
            "deterministic_optimizer_replay": "NOT_RUN",
            "allocation_construction": "held_dinic_partial_plus_lp_positive_carrier_flows",
            "allocation_solver": "held_dinic_partial_plus_sparse_lp_positive_carrier_flow",
            "cost_optimality_certified": False,
            "semantic_candidate_edge_count_basis": "pinned held+LP proposal graph count; feasibility does not reconstruct optimization graph",
            "augmentation_count_basis": "held partial Dinic subflow only; LP construction is counted separately",
        }
        for key, expected in required.items():
            if receipt.get(key) != expected:
                raise ValueError(f"LP-feasibility receipt has invalid {key}")
        held_aug = receipt.get("held_dinic_augmentation_count")
        if not isinstance(held_aug, int) or held_aug <= 0 or receipt.get("augmentations") != held_aug:
            raise ValueError("LP-feasibility receipt lacks truthful held Dinic augmentation count")
        for key in ("lp_positive_variable_count", "held_positive_flow_rows", "held_candidate_edges", "lp_candidate_edges"):
            if not isinstance(receipt.get(key), int) or receipt[key] <= 0:
                raise ValueError(f"LP-feasibility receipt lacks positive {key}")
        cap = receipt.get("maximum_allowed_candidate_edges")
        candidate_edges = receipt.get("candidate_edges")
        if not isinstance(cap, int) or cap <= 0 or not isinstance(candidate_edges, int) or not (0 < candidate_edges <= cap):
            raise ValueError("LP-feasibility receipt candidate graph count exceeds or lacks its declared budget")
        if receipt["held_candidate_edges"] + receipt["lp_candidate_edges"] > cap:
            raise ValueError("LP-feasibility receipt held and LP candidate graph counts exceed declared budget")
        if "recorded_flow_deterministic_replay" in receipt and receipt["recorded_flow_deterministic_replay"] == "PASS":
            raise ValueError("LP-feasibility receipt must not claim deterministic optimizer replay")
        pins = receipt.get("input_sha256")
        if not isinstance(pins, dict):
            raise ValueError("LP-feasibility receipt has no input_sha256 map")
        for key in ("input_child_node_sha256", "input_child_element_sha256",
                    "input_parent_mass_csv_sha256", "input_parent_label_csv_sha256", "support_binary_sha256",
                    "input_region_label_sha256"):
            if key not in pins:
                raise ValueError(f"LP-feasibility receipt misses input pin {key}")
            _sha(pins[key], f"input_sha256.{key}")
            if receipt.get(key) != pins[key]:
                raise ValueError(f"LP-feasibility flat/nested input pin differs: {key}")
        for key in ("child_mesh_sha256", "semantic_support_sha256", "transport_receipt_sha256",
                    "admissibility_source_support_sha256", "input_parent_mass_sha256",
                    "input_parent_source_sha256", "input_coordinate_frame_sha256",
                    "source_carrier_packet_sha256", "source_carrier_array_sha256",
                    "source_carrier_flow_trace_sha256", "feasibility_source_sha256", "lp_proposal_sha256",
                    "lp_input_manifest_sha256", "held_partial_manifest_sha256",
                    "held_carrier_flow_csv_sha256"):
            _sha(receipt.get(key), key)
        if receipt.get("admissibility_max_source_region_hops") != 1 or not isinstance(receipt.get("admissible_region_slots"), int):
            raise ValueError("LP-feasibility receipt has invalid source-region support declaration")
        flow_csv = receipt.get("flow_csv")
        if not isinstance(flow_csv, dict) or set(flow_csv) != FLOW_NAMES:
            raise ValueError("LP-feasibility receipt does not pin the exact six-CSV contract")
        for name, meta in flow_csv.items():
            if not isinstance(meta, dict):
                raise ValueError(f"invalid flow CSV pin: {name}")
            _sha(meta.get("sha256"), f"flow_csv.{name}.sha256")
            if not isinstance(meta.get("size_bytes"), int) or meta["size_bytes"] < 0:
                raise ValueError(f"invalid flow CSV size: {name}")
        return "solver_free_lp_feasibility"
    if status == f"PASS_SOURCE_ONLY_{role_upper}_{LEGACY_SUFFIX}":
        if receipt.get("recorded_flow_deterministic_replay") != "PASS":
            raise ValueError("legacy allocation receipt lacks deterministic replay PASS")
        if receipt.get("physical_admission") is not False or receipt.get("native_steps") != 0:
            raise ValueError("legacy allocation receipt claims physical/native admission")
        if set(receipt.get("flow_csv", {})) != FLOW_NAMES:
            raise ValueError("legacy allocation receipt does not pin exact six-CSV contract")
        return "deterministic_optimizer_replay"
    raise ValueError(f"unsupported or mismatched {role} source-only allocation status")

def validate_current_input_pins(receipt: dict, expected: dict[str, str]) -> None:
    """Check actual file digests against the flat fields and nested map in the receipt."""
    pins = receipt.get("input_sha256")
    if not isinstance(pins, dict):
        raise ValueError("receipt input_sha256 is not an object")
    file_keys = {
        "input_child_node_sha256", "input_child_element_sha256", "input_parent_mass_csv_sha256",
        "input_parent_label_csv_sha256", "support_binary_sha256", "input_region_label_sha256",
    }
    for key, digest in expected.items():
        _sha(digest, f"expected.{key}")
        if receipt.get(key) != digest:
            raise ValueError(f"receipt input pin mismatch: {key}")
        if key in file_keys and pins.get(key) != digest:
            raise ValueError(f"receipt nested input file pin mismatch: {key}")
    parent_mass = expected.get("input_parent_mass_csv_sha256")
    parent_source = expected.get("input_parent_source_sha256")
    if parent_mass and receipt.get("input_parent_mass_sha256") != parent_mass:
        raise ValueError("parent mass packet hash does not match compiled mass CSV")
    if parent_source and receipt.get("input_parent_source_sha256") != parent_source:
        raise ValueError("parent source contribution hash does not match compiled source CSV")
