import importlib.util
import hashlib
import io
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/audit_cardiac_case18_source_fields.py"
SPEC = importlib.util.spec_from_file_location("case18_source_fields", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def vtk_fixture(extra_field: bytes = b"") -> bytes:
    lines = [
        b"DATASET UNSTRUCTURED_GRID\n",
        b"POINTS 300965 float\n",
        b"CELL_TYPES 1470083\n",
        b"CELLS 1470083 7350415\n",
        b"CELL_DATA 1470083\n",
        b"SCALARS ID int 1\n",
        b"VECTORS fibres float\n",
        b"VECTORS sheets float\n",
        b"POINT_DATA 300965\n",
        b"SCALARS RHO.dat float 1\n",
        b"SCALARS PHI.dat float 1\n",
        b"SCALARS Z.dat float 1\n",
        b"SCALARS V.dat float 1\n",
    ]
    if extra_field:
        lines.append(extra_field + b"\n")
    return b"".join(lines)


def test_inventory_captures_declared_source_fields_and_member_hash():
    source = vtk_fixture()
    structure, declarations, digest, byte_count = audit.inspect_vtk(io.BytesIO(source))
    audit.verify_declarations(structure, declarations)
    assert structure == audit.EXPECTED_STRUCTURE
    assert declarations == audit.EXPECTED_DECLARATIONS
    assert digest == hashlib.sha256(source).hexdigest()
    assert byte_count == len(source)


@pytest.mark.parametrize("field", [
    b"SCALARS activation_time_ms float 1",
    b"SCALARS FEC_cell_tag int 1",
    b"FIELD FieldData 1",
])
def test_unlisted_electrical_or_field_data_cannot_be_claimed_absent(field):
    structure, declarations, _, _ = audit.inspect_vtk(io.BytesIO(vtk_fixture(field)))
    with pytest.raises(ValueError, match="unexpected or missing source field"):
        audit.verify_declarations(structure, declarations)
