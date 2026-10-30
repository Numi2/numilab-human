#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
NEW=Path(__file__).resolve().parent/'prepare_typed_connected_union.py'
OLD=ROOT/'quality-warning-connected-union-037-typed-metadata-001'/'prepare_warning_union_typed_metadata.py'
OPDIR=ROOT/'quality-warning-connected-union-037'/'run-002'

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

old=load_module(OLD,'old037')
new=load_module(NEW,'generic038')
summary={}
for role in ('O','I','D'):
    expected=old.operation_files(OPDIR,role,'037')
    actual=new.operation_files(OPDIR,role,'037')
    assert set(expected)==set(actual), (role, sorted(expected), sorted(actual))
    for operation_id in expected:
        old_data,old_path,old_sha=expected[operation_id]
        new_data,new_path,new_sha=actual[operation_id]
        assert old_path==new_path and old_sha==new_sha
        for key,value in old_data.items():
            assert new_data.get(key)==value, (role,operation_id,key)
        assert new_data['retired_input_face_rows']==old_data['retired_input_face_rows']
        assert new_data['new_faces_input_point_rows']==old_data['new_faces_input_point_rows']
    summary[role]={'operation_count':len(actual),'operation_ids':sorted(actual)}
print(json.dumps({'status':'PASS_037_DEFAULT_OPERATION_FILE_PARITY','summary':summary},sort_keys=True))
