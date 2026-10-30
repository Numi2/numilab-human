#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, hashlib, json, tempfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('typed_connected_union', HERE/'prepare_typed_connected_union.py')
M=importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(M)

def write_npz(path:Path, **arrays): np.savez_compressed(path, **arrays)

def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

with tempfile.TemporaryDirectory(prefix='typed038-zero-op-') as td:
    root=Path(td)
    # A tiny source/typed/immediate-parent surface with 45 I points for the existing
    # protected-route cardinality contract. This exercises metadata logic only.
    theta=np.linspace(0.0,2*np.pi,44,endpoint=False)
    points=np.vstack((np.array([[0.0,0.0,0.25]]),np.column_stack((np.cos(theta),np.sin(theta),np.zeros_like(theta))))).astype(np.float64)
    faces=np.asarray([[0,i,i+1] for i in range(1,44)],dtype=np.int64)
    source_sha='1'*64
    parent_raw_path=root/'parent-037-raw.npz'
    write_npz(parent_raw_path,points_world_m=points,points_world_m_f32=points.astype(np.float32),faces=faces)
    parent_raw_sha=sha(parent_raw_path)
    chain={'original':{'sha256':source_sha},'chain':[{'path':str(parent_raw_path),'sha256':parent_raw_sha}]}
    npoint=len(points); nface=len(faces)
    parent={
        'points_world_m':points.copy(), 'points_world_m_f32':points.astype(np.float32), 'faces':faces.copy(),
        'face_source_layer':np.full(nface,2,np.int64), 'face_source_body':np.full(nface,23,np.int64),
        'face_source_bone':np.full(nface,31,np.int64), 'face_source_face_id':np.arange(nface,dtype=np.int64),
        'face_source_original_id':np.full(nface,51,np.int64), 'face_source_identity0':np.arange(nface,dtype=np.int64),
        'face_source_identity1':np.zeros(nface,np.int64),
        'face_source_identity_schema_by_original_id_json':np.asarray(['{}']),
        'face_ancestry_offsets':np.arange(nface+1,dtype=np.int64),
        'face_ancestry_source_face_rows':np.arange(nface,dtype=np.int64),
        'face_ancestry_source_original_ids':np.full(nface,51,np.int64),
        'face_ancestry_parent_face_rows':np.arange(nface,dtype=np.int64),
        'quality_star_point_output_to_input_row':np.arange(npoint,dtype=np.int64),
        'quality_star_face_output_to_input_row':np.arange(nface,dtype=np.int64),
        'quality_star_map_chain_json':np.asarray([M.canonical_json(chain)]),
    }
    parent_typed_path=root/'parent-037-typed.npz'; write_npz(parent_typed_path,**parent)
    maps_path=root/'parent-037-maps.npz'
    write_npz(maps_path,raw_output_point_to_compact_row=np.arange(npoint,dtype=np.int64),output_face_to_raw_row=np.arange(nface,dtype=np.int64))
    proposal_path=root/'proposal-038-identity.npz'
    proposal={
        'points_world_m':points.copy(),'points_world_m_f32':points.astype(np.float32),'faces':faces.copy(),
        'point_output_to_parent_rows':np.arange(npoint,dtype=np.int64),
        'face_output_to_parent_rows':np.arange(nface,dtype=np.int64),
        'point_output_to_input_rows':np.arange(npoint,dtype=np.int64),
        'face_output_to_input_rows':np.arange(nface,dtype=np.int64),
        'immediate_parent_candidate_sha256':np.asarray([parent_raw_sha]),
        'source_candidate_sha256':np.asarray([source_sha]),
        'candidate_patch_operation_ids':np.asarray([],dtype='<U1'),
    }
    write_npz(proposal_path,**proposal)
    receipt_routes=[]
    for role,count in (('O',45),('I',45),('D',59)):
        for local in range(count):
            canonical=len(receipt_routes)
            receipt_routes.append({'canonical_protected_row':canonical,'role':role,'surface_vertex_row':local if role=='I' else 0,'target_world_m_f32':(points[local].astype(np.float32).tolist() if role=='I' else [0.0,0.0,0.0])})
    protected_path=root/'protected.json'
    protected_path.write_text(json.dumps({'protected149':{'count':149,'surface_routes':receipt_routes}}))
    out=root/'out'
    out.mkdir()
    result=M.build_role('I',parent_typed_path,maps_path,parent_raw_path,proposal_path,root,proposal_path,protected_path,out,generation_label='038',operation_glob='{stage}-{role}-union-*.json')
    with np.load(out/'candidate-I-connected-union-typed.npz',allow_pickle=False) as z:
        assert np.array_equal(z['faces'],faces)
        assert np.array_equal(z['points_world_m'],points)
        assert np.array_equal(z['face_source_layer'],parent['face_source_layer'])
        assert np.array_equal(z['face_ancestry_offsets'],parent['face_ancestry_offsets'])
        assert np.array_equal(z['quality_star_immediate_face_output_to_input_row'],np.arange(nface))
        assert np.array_equal(z['quality_star_immediate_point_output_to_input_row'],np.arange(npoint))
    assert result['new_faces']==0 and result['patch_count']==0
    assert result['typed_patch_union_tuple_count']==nface
    assert result['protected_route_rows'] and len(result['protected_route_rows'])==45
    print('PASS zero-operation 038 typed build: geometry/maps retained exactly; inherited face ancestry nonempty; 45 I routes preserved')
