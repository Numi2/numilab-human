#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('typed_connected_union', HERE/'prepare_typed_connected_union.py')
M=importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(M)


def write_op(directory:Path, filename:str, *, role='O', patch_id='p0', face_key='retired_input_face_rows', point_key='new_faces_input_point_rows', nested=True):
    patch={'patch_id':patch_id,'new_output_face_rows':[2],'source_lineage':{}}
    if nested:
        patch[face_key]=[7]
        patch[point_key]=[[0,1,2]]
    data={'role':role,'patch_id':patch_id,'new_output_face_rows':[2],face_key:[7],point_key:[[0,1,2]],'source_lineage':{},'patches':[patch]}
    p=directory/filename
    p.write_text(json.dumps(data))
    return p


def expect_error(fn, needle):
    try: fn()
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
    else: raise AssertionError(f'expected ValueError containing {needle!r}')

with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    # Stable generic contract for 038: filenames are configurable and row domains are explicit.
    d=root/'generic'; d.mkdir()
    write_op(d,'038-O-patch-000.json')
    got=M.operation_files(d,'O','038',operation_glob='{stage}-{role}-patch-*.json')
    op=got['p0'][0]
    assert op['retired_input_face_rows']==[7]
    assert op['new_faces_input_point_rows']==[[0,1,2]]
    assert op['_resolved_operation_retired_face_key']=='retired_input_face_rows'
    assert op['_resolved_operation_new_point_key']=='new_faces_input_point_rows'

    # Explicit parent key parameters support a producer-specific spelling without renaming its JSON.
    c=root/'custom'; c.mkdir()
    write_op(c,'038-O-custom-000.json',face_key='retired_037_rows',point_key='triangles_037_points')
    got=M.operation_files(c,'O','038',operation_glob='{stage}-{role}-custom-*.json',operation_retired_face_key='retired_037_rows',operation_new_point_key='triangles_037_points')
    op=got['p0'][0]
    assert op['retired_input_face_rows']==[7] and op['new_faces_input_point_rows']==[[0,1,2]]

    # Legacy 037 names remain the default and normalize to the same internal row-domain fields.
    legacy=root/'legacy'; legacy.mkdir()
    write_op(legacy,'037-O-warning-000.json',face_key='source_036_face_rows',point_key='new_faces_036_point_rows',nested=False)
    got=M.operation_files(legacy,'O','037')
    op=got['p0'][0]
    assert op['retired_input_face_rows']==[7] and op['new_faces_input_point_rows']==[[0,1,2]]
    assert op['_resolved_operation_retired_face_key']=='source_036_face_rows'

    # New generic rows must agree between operation top-level and nested patch if both are present.
    bad=root/'bad'; bad.mkdir()
    p=write_op(bad,'038-O-patch-000.json')
    data=json.loads(p.read_text()); data['patches'][0]['retired_input_face_rows']=[8]; p.write_text(json.dumps(data))
    expect_error(lambda:M.operation_files(bad,'O','038',operation_glob='{stage}-{role}-patch-*.json'),'top/nested retired')

    # Missing selected key fails closed; it never guesses a differently named row domain.
    missing=root/'missing'; missing.mkdir()
    write_op(missing,'038-O-patch-000.json',face_key='retired_015_rows',point_key='new_faces_015_points',nested=False)
    expect_error(lambda:M.operation_files(missing,'O','038',operation_glob='{stage}-{role}-patch-*.json'),'lacks configured')

print('PASS generic 038 immediate-parent row keys; 037 default compatibility; custom keys and fail-closed mismatches')
