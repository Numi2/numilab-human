import struct

import pytest

from numilab_human.open_knee_xplt import inspect_contact_states


def chunk(tag, data):
    return struct.pack('<II', tag, len(data)) + data


def integer(tag, value):
    return chunk(tag, struct.pack('<I', value))


def plot(*, compression=0, width=2):
    dictionary = b''
    for name in ('contact gap', 'contact pressure'):
        dictionary += chunk(0x01020001, integer(0x01020002, 0) + integer(0x01020003, 1)
                            + chunk(0x01020004, name.encode() + b'\0'))
    header = chunk(0x01010000, integer(0x01010001, 5) + integer(0x01010004, compression))
    surface = chunk(0x01043100, chunk(0x01043101, integer(0x01043102, 1)
        + integer(0x01043103, width) + chunk(0x01043104, struct.pack('<I', 3) + b'PTC')))
    root = chunk(0x01000000, header + chunk(0x01020000, chunk(0x01025000, dictionary))
        + chunk(0x01040000, chunk(0x01043000, surface)))
    fields = b''
    for i, values in enumerate(([-.002, .001], [0., .25]), 1):
        fields += chunk(0x02020001, integer(0x02020002, i) + chunk(0x02020003,
            chunk(1, struct.pack('<ff', *values))))
    state = chunk(0x02000000, chunk(0x02010000, chunk(0x02010002, struct.pack('<f', .05)))
        + chunk(0x02020000, chunk(0x02020500, fields)))
    return b'BEF\0' + root, state


def test_contact_reader_preserves_face_counts_units_and_pressure_support(tmp_path):
    root, state = plot()
    p = tmp_path/'source.xplt'; p.write_bytes(root+state)
    r = inspect_contact_states(p, expected_bytes=p.stat().st_size)
    assert r['archive_complete']
    fields = r['states'][0]['contact_fields']
    assert fields[0]['negative_faces'] == 1
    assert fields[0]['units'] == 'mm'
    assert fields[1]['positive_faces'] == 1
    assert fields[1]['maximum'] == .25
    assert fields[1]['units'] == 'MPa'


def test_partial_download_exposes_only_complete_states(tmp_path):
    root, state = plot()
    p = tmp_path/'source.xplt.partial'; p.write_bytes(root+state+state[:19])
    r = inspect_contact_states(p, expected_bytes=len(root)+2*len(state))
    assert not r['archive_complete']
    assert r['complete_chunk_prefix_bytes'] == len(root)+len(state)
    assert len(r['states']) == 1


def test_complete_wrong_archive_identity_rejected(tmp_path):
    root, state = plot()
    p = tmp_path/'source.xplt';p.write_bytes(root+state)
    with pytest.raises(ValueError, match='identity drift'):
        inspect_contact_states(p, expected_bytes=p.stat().st_size, expected_sha256='0'*64)


@pytest.mark.parametrize('options', [{'compression': 1}, {'width': 3}])
def test_unsupported_encoding_and_wrong_face_layout_fail(tmp_path, options):
    root, state = plot(**options)
    p = tmp_path/'source.xplt';p.write_bytes(root+state)
    with pytest.raises(ValueError):
        inspect_contact_states(p)
