import json
import hashlib
import struct

import pytest

from numilab_human.open_knee_xplt import (
    compare_source_checkpoint_arrays,
    inspect_contact_states,
)


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


def full_source_plot():
    dictionaries = {}
    for tag, names in (
        (0x01023000, ('displacement', 'reaction forces')),
        (0x01024000, ('stress', 'rigid position')),
        (0x01025000, ('contact gap', 'contact pressure', 'contact traction')),
    ):
        fields = b''
        for name in names:
            if name in ('stress',):
                kind, fmt = 2, 1
            elif name == 'rigid position' or name == 'contact traction':
                kind, fmt = 1, 3 if name == 'rigid position' else 1
            else:
                kind, fmt = 0 if name in ('contact gap', 'contact pressure') else 1, 0
                if name in ('contact gap', 'contact pressure'):
                    fmt = 1
            fields += chunk(0x01020001, integer(0x01020002, kind) + integer(0x01020003, fmt)
                            + chunk(0x01020004, name.encode() + b'\0'))
        dictionaries[tag] = chunk(tag, fields)
    dictionary = chunk(0x01020000, b''.join(dictionaries.values()))
    header = chunk(0x01010000, integer(0x01010001, 5) + integer(0x01010004, 0))
    parts = chunk(0x01030000, chunk(0x01030001,
        integer(0x01030002, 4) + chunk(0x01030003, b'Femur\0')))
    nodes = chunk(0x01041000, chunk(0x01041001, struct.pack('<6f', 0, 0, 0, 1, 0, 0)))
    domain = chunk(0x01042100, chunk(0x01042101,
        integer(0x01042102, 2) + integer(0x01042103, 4)
        + integer(0x01032104, 1)
        + chunk(0x01032105, struct.pack('<I', 3) + b'FEM'))
        + chunk(0x01042200, b''))
    domains = chunk(0x01042000, domain)
    surface = chunk(0x01043100, chunk(0x01043101, integer(0x01043102, 1)
        + integer(0x01043103, 2) + chunk(0x01043104, struct.pack('<I', 3) + b'PTC')))
    surfaces = chunk(0x01043000, surface)
    root = chunk(0x01000000, header + dictionary + parts
                 + chunk(0x01040000, nodes + domains + surfaces))

    def variable(identifier, data_chunks):
        return chunk(0x02020001, integer(0x02020002, identifier)
                     + chunk(0x02020003, b''.join(data_chunks)))

    nodal = chunk(0x02020300,
        variable(1, [chunk(0, struct.pack('<6f', 1, 2, 3, 4, 5, 6))])
        + variable(2, [chunk(0, struct.pack('<6f', 7, 8, 9, 10, 11, 12))]))
    element = chunk(0x02020400,
        variable(1, [chunk(1, struct.pack('<6f', 13, 14, 15, 16, 17, 18))])
        + variable(2, [chunk(1, struct.pack('<3f', 19, 20, 21))]))
    face = chunk(0x02020500,
        variable(1, [chunk(1, struct.pack('<2f', -.01, .02))])
        + variable(2, [chunk(1, struct.pack('<2f', 0, .3))])
        + variable(3, [chunk(1, struct.pack('<6f', 1, 2, 3, 4, 5, 6))]))
    state = chunk(0x02000000,
        chunk(0x02010000, chunk(0x02010002, struct.pack('<f', .25)))
        + chunk(0x02020000, nodal + element + face))
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


def test_source_reader_maps_full_mechanical_fields_and_exports_checkpoint(tmp_path):
    root, state = full_source_plot()
    plot_path = tmp_path/'source.xplt'
    plot_path.write_bytes(root+state)
    output = tmp_path/'checkpoints'
    result = inspect_contact_states(plot_path, expected_bytes=plot_path.stat().st_size,
        include_source_fields=True, checkpoint_states=(0,), checkpoint_output=output)
    assert result['archive_complete']
    assert result['node_count'] == 2
    assert result['domains'][0]['part_name'] == 'Femur'
    assert len(result['states'][0]['source_fields']) == 7
    fields = result['states'][0]['source_fields']
    lookup = {(x['class'], x['field'], x.get('domain_id'), x.get('surface_id')): x for x in fields}
    assert lookup[('node', 'displacement', None, None)]['rms'] > 0
    assert lookup[('element', 'stress', 1, None)]['components_per_item'] == 6
    assert lookup[('element', 'rigid position', 1, None)]['units'] == 'mm'
    assert next(x for x in result['states'][0]['contact_fields']
                if x['field'] == 'contact gap')['negative_faces'] == 1
    assert lookup[('surface', 'contact traction', None, 1)]['components_per_item'] == 3
    assert result['states'][0]['checkpoint_array_archive_sha256'] == hashlib.sha256(
        (output/'state-00000.npz').read_bytes()).hexdigest()
    import numpy as np
    with np.load(output/'state-00000.npz') as arrays:
        np.testing.assert_array_equal(arrays['node_01'], [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(arrays['element_01_domain_01'], [13, 14, 15, 16, 17, 18])
        assert 'metadata_json' in arrays


def test_checkpoint_export_is_immutable(tmp_path):
    root, state = full_source_plot()
    plot_path = tmp_path/'source.xplt'; plot_path.write_bytes(root+state)
    output = tmp_path/'checkpoints'
    inspect_contact_states(plot_path, expected_bytes=plot_path.stat().st_size,
        include_source_fields=True, checkpoint_states=(0,), checkpoint_output=output)
    with pytest.raises(FileExistsError):
        inspect_contact_states(plot_path, expected_bytes=plot_path.stat().st_size,
            include_source_fields=True, checkpoint_states=(0,), checkpoint_output=output)


def test_matter_checkpoint_comparison_checks_identity_layout_and_reports_error(tmp_path):
    root, state = full_source_plot()
    plot_path = tmp_path/'source.xplt'; plot_path.write_bytes(root+state)
    output = tmp_path/'checkpoints'
    inspect_contact_states(plot_path, expected_bytes=plot_path.stat().st_size,
        include_source_fields=True, checkpoint_states=(0,), checkpoint_output=output)
    import numpy as np
    with np.load(output/'state-00000.npz') as archive:
        arrays = {key: np.array(archive[key], copy=True)
                  for key in archive.files if key != 'metadata_json'}
        metadata = json.loads(str(archive['metadata_json'].item()))
    metadata['reference_plot_sha256'] = 'a' * 64
    reference = tmp_path/'reference.npz'
    matter = tmp_path/'matter.npz'
    np.savez_compressed(reference, metadata_json=json.dumps(metadata), **arrays)
    np.savez_compressed(matter, metadata_json=json.dumps(metadata), **arrays)
    result = compare_source_checkpoint_arrays(reference, matter)
    assert result['status'] == 'compared'
    assert all(item['rms_error'] == 0 for item in result['arrays'])

    arrays['node_01'][0] += 1
    np.savez_compressed(matter, metadata_json=json.dumps(metadata), **arrays)
    result = compare_source_checkpoint_arrays(reference, matter)
    changed = next(item for item in result['arrays'] if item['array'] == 'node_01')
    assert changed['maximum_absolute_error'] == 1
    assert changed['mean_bias'] == pytest.approx(1 / 6)

    metadata['continuation_time'] += 0.01
    np.savez_compressed(matter, metadata_json=json.dumps(metadata), **arrays)
    with pytest.raises(ValueError, match='same FEBio continuation state'):
        compare_source_checkpoint_arrays(reference, matter)


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
