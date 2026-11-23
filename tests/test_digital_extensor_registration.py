"""Exercise the bilateral registration against a paired native source artifact."""
import os
from pathlib import Path

import pytest

from numilab_human import model


def test_only_the_eight_named_insertions_move(tmp_path):
    reference = os.environ.get("NUMILAB_HUMAN_NATIVE_REFERENCE_ARTIFACT")
    bones = os.environ.get("NUMILAB_HUMAN_MOTION_BONES")
    if not reference or not bones:
        pytest.skip("requires the paired MyoSim and registered BodyParts3D artifacts")

    def build(name, enabled):
        result = model.numi_human_tendon_attachment_envelope_payload(
            Path(reference), Path(bones), tmp_path / name,
            register_digital_extensor_insertions=enabled,
        )
        records = result["endpoints"]
        keyed = {(row["muscle"], row["endpoint"]): row for row in records}
        assert len(keyed) == len(records)
        return keyed

    control = build("control", False)
    registered = build("registered", True)
    expected_members = {
        "EDC2": "FJ3193", "EDC2_l": "FJ3183", "EDC3": "FJ3196", "EDC3_l": "FJ3186",
        "EDC4": "FJ3197", "EDC4_l": "FJ3187", "EDC5": "FJ3194", "EDC5_l": "FJ3184",
    }
    assert control.keys() == registered.keys()
    changed = {key for key in control if control[key] != registered[key]}
    assert changed == {(name, "insertion") for name in expected_members}
    for name, member in expected_members.items():
        row = registered[name, "insertion"]
        source = control[name, "insertion"]
        for key in ("body_index", "route_node_index", "source_site_index", "source_local_point_m"):
            assert row[key] == source[key]
        assert row["surface"]["bone_member_id"] == member
        assert row["attachment_mode"] == "registered_bone_migrated_distributed_envelope"
        assert row["resolved_local_point_m"] == pytest.approx(row["surface"]["nearest_local_point_m"])
        assert 0 < row["endpoint_migration_m"] <= 0.012
