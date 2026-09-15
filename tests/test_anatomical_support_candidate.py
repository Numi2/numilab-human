import hashlib
import struct
from pathlib import Path

import pytest

from numilab_human.anatomical_support_candidate import (
    CONTACT_HEADER,
    NHCNT1_RECORD,
    NHBONES_HEADER,
    NHBONES_RECORD,
    NHRIGID_HEADER,
    compile_anatomical_support_candidate,
)
from numilab_human.model import ImportError


GEOMETRIES = [
    (408, 153), (377, 139), (409, 153), (378, 139), (405, 152),
    (374, 138), (406, 152), (375, 138), (407, 152), (376, 138),
]


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_hash = bytes.fromhex("ab" * 32)
    support = CONTACT_HEADER.pack(
        b"NHCNT1\0\0", 1, 157, len(GEOMETRIES), 0, source_hash,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    )
    for index, (geometry, body) in enumerate(GEOMETRIES):
        point = (0.01 * index, -0.002 * index, -0.03)
        support += NHCNT1_RECORD.pack(
            body, geometry, *point, 0.0, 0.0, 0.0, 1.0, 0.05 + index * 1.0e-4, 0.0, 0.0
        )
    bones = NHBONES_HEADER.pack(b"NHBONES1", 2, 4, 0, 0, 123, source_hash)
    quaternions = {
        138: (-0.53968447, 0.50425023, -0.49565917, 0.45694017),
        139: (-0.64160615, 0.32212242, -0.64935017, 0.25084466),
        152: (-0.50468016, 0.53928244, -0.45733440, 0.49529549),
        153: (-0.32263342, 0.64134902, -0.25136146, 0.64915061),
    }
    for stable_id, body in enumerate((138, 139, 152, 153), start=1):
        bones += NHBONES_RECORD.pack(body, 0, 0, 0, 0, stable_id, 0.0, 0.0, 0.0,
                                     *quaternions[body], 1.0)
    rigid = NHRIGID_HEADER.pack(
        b"NHRIGID2", 1, 1, 153, 157, 156, 157, 156, 0, 0, 0, source_hash,
    )
    paths = []
    for name, raw in (("support.nhcnt", support), ("bones.nhbones", bones),
                      ("rigid.nhrigid", rigid)):
        path = tmp_path / name
        path.write_bytes(raw)
        paths.append(path)
    return tuple(paths)  # type: ignore[return-value]


def test_candidate_is_deterministic_and_source_bound(tmp_path: Path) -> None:
    support, bones, rigid = _fixtures(tmp_path)
    first, payload = compile_anatomical_support_candidate(support, bones, rigid)
    second, payload_again = compile_anatomical_support_candidate(support, bones, rigid)
    assert payload == payload_again
    assert first == second
    assert first["payload"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert first["counts"] == {"support_bodies": 4, "source_witnesses": 10, "candidate_primitives": 10}
    assert first["qualification"]["anatomical_support_surface_candidate"] is True


@pytest.mark.parametrize("which", ["support", "bones", "rigid"])
def test_foreign_source_hash_rejected(tmp_path: Path, which: str) -> None:
    support, bones, rigid = _fixtures(tmp_path)
    path = {"support": support, "bones": bones, "rigid": rigid}[which]
    raw = bytearray(path.read_bytes())
    if which == "support":
        raw[20] ^= 0x01
    elif which == "bones":
        raw[NHBONES_HEADER.size - 1] ^= 0x01
    else:
        raw[NHRIGID_HEADER.size - 1] ^= 0x01
    path.write_bytes(raw)
    with pytest.raises(ImportError):
        compile_anatomical_support_candidate(support, bones, rigid)


def test_unreviewed_geometry_is_rejected(tmp_path: Path) -> None:
    support, bones, rigid = _fixtures(tmp_path)
    raw = bytearray(support.read_bytes())
    struct.pack_into("<I", raw, CONTACT_HEADER.size + 4, 999)
    support.write_bytes(raw)
    with pytest.raises(ImportError, match="unknown source geometry"):
        compile_anatomical_support_candidate(support, bones, rigid)
