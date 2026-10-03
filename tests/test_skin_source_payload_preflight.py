"""Fail-closed parsing of the ABI 5 skin source payload."""
import struct

import numpy as np
import pytest

from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.model import ImportError as HumanImportError


def _payload(magic=b'NHSKIN1\0', abi=5):
    binding_count, vertex_count, index_count = 1, 3, 3
    header = struct.pack('<8s5I32s', magic, abi, binding_count, vertex_count,
                         index_count, 0x12345678, bytes(range(32)))
    bindings = np.zeros((binding_count, 9), dtype='<f4').tobytes()
    vertices = np.zeros((vertex_count, 14), dtype='<f4')
    vertices[:, 6] = 0
    vertices[:, 10] = 1
    indices = np.asarray([0, 1, 2], dtype='<u4').tobytes()
    full_weights = np.ones((vertex_count, binding_count), dtype='<f4').tobytes()
    return header + bindings + vertices.tobytes() + indices + full_weights


def test_decode_abi5_skin_payload_sections():
    decoded = decode_payload(_payload())
    assert decoded['binding_count'] == 1
    assert decoded['vertex_count'] == 3
    assert decoded['index_count'] == 3
    assert decoded['registration_fingerprint32'] == 0x12345678
    assert decoded['source_archive_sha256'] == bytes(range(32)).hex()
    assert decoded['indices'].tolist() == [0, 1, 2]
    assert decoded['full_weights'].shape == (3, 1)
    assert decoded['full_weights'].sum(axis=1).tolist() == [1, 1, 1]


@pytest.mark.parametrize('raw, message', [
    (b'bad', 'truncated payload header'),
    (_payload(magic=b'BROKEN\0\0'), 'payload magic/ABI'),
    (_payload(abi=4), 'payload magic/ABI'),
    (_payload()[:-4], 'payload byte length'),
])
def test_decode_rejects_ambiguous_or_truncated_payload(raw, message):
    with pytest.raises(HumanImportError, match=message):
        decode_payload(raw)
