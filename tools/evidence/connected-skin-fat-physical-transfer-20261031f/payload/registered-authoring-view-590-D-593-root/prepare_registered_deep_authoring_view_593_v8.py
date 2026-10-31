#!/usr/bin/env python3
"""Versioned authoring-only D adapter layered on the immutable 590 helper.

This changes only authoring tuple/schema copies. It never edits physical D,
volume, O/I, registration, or published files. Derived role-D tuples are
projected to the D authoring layer while their source-layer ancestry stays
byte-for-byte in the provenance archive.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, sys
from pathlib import Path
from typing import Any
import numpy as np

INPUT_SHA256='2d7d4a3af13122aed0a7e44cfae0a4afd23508cd8eb14404e341c17e79274147'
REGISTRY_SHA256='ff73291bce77dda81eff985fecdbba1b198f24e95a7f8f9af1d6058172d4da45'
FILTERED_SOURCE_SHA256='d72f706524329f9b7a6390e4da273f49089fa809b1adbd6bdb3974d6af670cb5'
RING1_SHA256='8244061e75f45456723b7bc7c6c36c6b08462801f8999ec86631d45baa6a5af0'
PIN_590_HELPER_SHA256='7cbf1b8365ba28956ad739eee0cd3752a188cc584c91316b331f5fbb39ec8e21'
PIN_230_EXPORTER_SHA256='18fbc383044c18f29d7fd7dccf822cc5b65957bc75aca0f4b6ac964ea27cb74a'
PIN_RING1_EXPECTED_PATH='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/paired-head-reference-fit-684/run-055-D-seam-disk-remesh/attempt-006/candidate-ring1-typed.npz'
PIN_CHILD_READER=('/Users/numi/Work/skin-fat-native-integration-209/apps/NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp','e2664c07868b2ee516ed543def59e5cb5e1ecf8d050110ce7ad4ec6d996b2d3c')
PIN_ROLE_CONSUMER=('/Users/numi/Work/skin-fat-native-integration-209/apps/NumiHumanPhysicalSkinAuthoringRoleConsumer.hpp','233fad8d034e39aa61f60685fb6eabb8f1f4e525f04ea06a789cba6d25e24496')
TUPLE=('layer','body','bone','face_id','original_id','identity0','identity1')
DERIVED_ID=62000204
D202_ID=62000202
D202_LINEAGE_SOURCE='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-006-emitted-face-lineage/D-derived-62000202-typed-lineage-source.npz'
D202_LINEAGE_SHA='9fe4046ccdfa2663b60d032666a3f57ad5c3193671b873432370230981bdd5d3'
D202_PARENT_F32='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-006-emitted-face-lineage/parent-run003-candidate-f32.npz'
D202_PARENT_F32_SHA='ed7017881078cf7784df42e9bdfbea69db40fb2a76ae57f2bed4f691a1a04225'
D202_LINEAGE_REPORT='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-006-emitted-face-lineage/report.json'
D202_LINEAGE_REPORT_SHA='4f1d61f25f6319e3284169a8d3882a4b31e297ff92d161cc132f77a1fe608227'
CENSUS_PATH='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/registered-authoring-view-590-D-593-root/run-007-D-source-metadata-census.json'
CENSUS_SHA='9b28e763937059c69f97612551b7ae963bdfa3292ec227bf743572c294c178ea'
TEMP_CAP_SOURCE='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-038-run037-fat-occupancy/closed-inputs/D-closed-run037-f32.npz'
TEMP_CAP_SOURCE_SHA='0dfc1bd8b6705b2c884c5b816a62a4bbb71e69b2865a9eee849ce16690b2366b'
TEMP_CAP_GEOM='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-038-run037-fat-occupancy/D-geometry-run037-f32.npz'
TEMP_CAP_GEOM_SHA='073db24d046703097150b97b9644b5d74e6f46c2ac5a79ace30dd8977f76f22f'
TEMP_CAP_RECEIPT='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-038-run037-fat-occupancy/D-geometry-receipt.json'
TEMP_CAP_RECEIPT_SHA='5640853bb0d6e1b6e064bbec917b8d20fd6c1be28363c02ff1e7c2a8ac442d94'
TEMP_CAP_PARENT='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-036-component21-convex-hull-50um/D-plus-component21-convex-hull-closed-f64.npz'
TEMP_CAP_PARENT_SHA='9d9a6488c4653a87fa90da933eb51fd3d9fb6ee34f6983ec3db6aa3061cac34d'
TEMP_CAP_PRODUCER='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-038-run037-fat-occupancy/audit_current_fat_run037_batched_resume.py'
TEMP_CAP_PRODUCER_SHA='67f7676bdf3f1a34959bf4e48bc5848e4182ceb5c0ae10104a54861ef8415931'
TEMP_CAP_PARENT_PRODUCER='/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/matched-nesting-390/run-036-component21-convex-hull-50um/build_component21_hull_union.py'
TEMP_CAP_PARENT_PRODUCER_SHA='03bee105f462d5e1ded6b7ea94ee2a7c4a838168760b321650f5234f8bb1a720'

class AdapterError(ValueError):
    pass

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def schema_text(z:dict[str,np.ndarray])->str:
    key='face_source_identity_schema_by_original_id_json'
    if key not in z or np.asarray(z[key]).shape!=(1,): raise AdapterError('missing singleton identity schema')
    return str(np.asarray(z[key]).ravel()[0])

def projection_ids(schema:dict[str,Any])->list[int]:
    """Project every active schema-declared derived D source into D authoring code 2."""
    return sorted(int(sid) for sid,info in schema.items()
        if info.get('role')=='D' and bool(info.get('derived_operation')))

def verify_layer3_provenance(candidate:dict[str,np.ndarray],schema:dict[str,Any])->dict[str,Any]:
    """Authenticate explicit source-code-3 D derivatives as inherited BONE ancestry."""
    ids=np.asarray(candidate['face_source_original_id'],dtype=np.int64)
    layers=np.asarray(candidate['face_source_layer'],dtype=np.int64)
    cids=np.asarray(candidate['quality_star_patch_union_face_source_original_id'],dtype=np.int64)
    clayers=np.asarray(candidate['quality_star_patch_union_face_source_layer'],dtype=np.int64)
    derived3=[];inactive3=[];parent_ids=set();primary_rows=0;csr_rows=0
    for sid_s,info in schema.items():
        if info.get('role')!='D' or not info.get('derived_operation') or info.get('layer')!=3:
            continue
        sid=int(sid_s)
        pr=np.flatnonzero(ids==sid); cr=np.flatnonzero(cids==sid)
        if len(pr)==0 and len(cr)==0:
            inactive3.append(sid)
            continue
        derived3.append(sid)
        if len(pr) and not np.all(layers[pr]==3): raise AdapterError(f'{sid}: primary tuple layer disagrees with pinned source layer 3')
        if len(cr) and not np.all(clayers[cr]==3): raise AdapterError(f'{sid}: CSR tuple layer disagrees with pinned source layer 3')
        primary_rows+=len(pr);csr_rows+=len(cr)
        parents=info.get('derived_parent_source_ids')
        if not isinstance(parents,list) or not parents: raise AdapterError(f'{sid}: layer-3 derived D source lacks parent IDs')
        for p in parents:
            p=int(p);pinfo=schema.get(str(p))
            if not isinstance(pinfo,dict) or pinfo.get('role')!='BONE':
                raise AdapterError(f'{sid}: source layer 3 parent {p} is not registered as BONE')
            p_rows=np.flatnonzero(ids==p)
            if len(p_rows)==0 or not np.all(layers[p_rows]==3):
                raise AdapterError(f'{sid}: BONE parent {p} has no matching layer-3 primary rows')
            parent_ids.add(p)
    return {'derived_D_layer3_ids':sorted(derived3),'derived_D_layer3_id_count':len(derived3),'inactive_registry_derived_D_layer3_ids':sorted(inactive3),
      'primary_tuple_rows':primary_rows,'csr_tuple_rows':csr_rows,
      'distinct_parent_source_ids':sorted(parent_ids),'distinct_parent_source_id_count':len(parent_ids),
      'all_layer3_parent_source_ids_registered_as_BONE':True,
      'all_parent_candidate_primary_tuples_verified_layer3':True,
      'semantics':'3 remains inherited BONE source provenance; parent registry role is BONE and its current candidate source tuples are layer 3; authoring role D is separately projected to layer code 2.'}

def _expected_source_codes(sid:int,info:dict[str,Any],schema:dict[str,Any],
                           primary_ids:np.ndarray,primary_layers:np.ndarray,
                           csr_ids:np.ndarray,csr_layers:np.ndarray)->set[int]:
    """Validate the original source-layer code(s) before authoring projection."""
    if info.get('layer') is not None:
        return {int(info['layer'])}
    parents=info.get('derived_parent_source_ids')
    if not isinstance(parents,list) or not parents:
        raise AdapterError(f'{sid}: derived D record has no source layer and no parent-source union')
    allowed=set()
    for parent in parents:
        parent=int(parent);pinfo=schema.get(str(parent))
        if not isinstance(pinfo,dict): raise AdapterError(f'{sid}: unlayered derived D record references missing parent source {parent}')
        prow=np.flatnonzero(primary_ids==parent);crow=np.flatnonzero(csr_ids==parent)
        observed=set(map(int,primary_layers[prow].tolist()))|set(map(int,csr_layers[crow].tolist()))
        if not observed:
            if pinfo.get('role')=='BONE': observed={3}
            elif pinfo.get('layer') is not None: observed={int(pinfo['layer'])}
            else: raise AdapterError(f'{sid}: parent source {parent} has no typed layer evidence')
        if pinfo.get('role')=='BONE' and observed!={3}:
            raise AdapterError(f'{sid}: BONE parent {parent} has non-BONE source layer tuples {sorted(observed)}')
        if not observed.issubset({-1,2,3}): raise AdapterError(f'{sid}: parent {parent} has unsupported source layer codes {sorted(observed)}')
        allowed.update(observed)
    if not allowed: raise AdapterError(f'{sid}: unlayered derived D record has no recoverable source-layer codes')
    return allowed

def audit_unlayered_derived_provenance(candidate:dict[str,np.ndarray],schema:dict[str,Any])->dict[str,Any]:
    ids=np.asarray(candidate['face_source_original_id'],dtype=np.int64);layers=np.asarray(candidate['face_source_layer'],dtype=np.int64)
    cids=np.asarray(candidate['quality_star_patch_union_face_source_original_id'],dtype=np.int64);clayers=np.asarray(candidate['quality_star_patch_union_face_source_layer'],dtype=np.int64)
    rows=[]
    for sid_s,info in schema.items():
        if info.get('role')!='D' or not info.get('derived_operation') or info.get('layer') is not None: continue
        sid=int(sid_s);pr=np.flatnonzero(ids==sid);cr=np.flatnonzero(cids==sid)
        actual=set(map(int,layers[pr].tolist()))|set(map(int,clayers[cr].tolist()))
        if not actual: continue
        allowed=_expected_source_codes(sid,info,schema,ids,layers,cids,clayers)
        if not actual.issubset(allowed): raise AdapterError(f'{sid}: source tuple layers {sorted(actual)} are not supported by parent role/layer union {sorted(allowed)}')
        rows.append({'id':sid,'parent_source_ids':info.get('derived_parent_source_ids'),
          'parent_role_supported_source_layers':sorted(allowed),'observed_primary_and_csr_source_layers':sorted(actual),
          'primary_rows':len(pr),'csr_rows':len(cr)})
    return {'unlayered_derived_D_count':len(rows),'records':rows,
      'semantics':'Mixed D/BONE parent unions retain their original tuple layer codes; the D authoring projection is a separate role namespace.'}

def verify_100000186(candidate:dict[str,np.ndarray],schema:dict[str,Any])->dict[str,Any]:
    """Bind the 13 identity rows to the pinned run010 source candidate and its proof."""
    sid=100000186;info=schema.get(str(sid))
    if not isinstance(info,dict) or info.get('role')!='D' or not info.get('derived_operation'):
        raise AdapterError(f'{sid}: canonical14 source is not a derived D record')
    droot=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/canonical-D-boundary-binding-696')
    run=droot/'run-010-combined-operational-reference-candidate'
    expected_source=run/'candidate-D-canonical14-reference-remap.npz'
    expected_source_sha='7d8419bdc4ad1d3892a946c93ad7f0f9b2db2e4302f8f666480612166e25de2a'
    if Path(info.get('path',''))!=expected_source or sha(expected_source)!=expected_source_sha:
        raise AdapterError(f'{sid}: canonical14 source path/hash differs from pinned run010 artifact')
    assets=[
      {'asset_role':'run010_input_candidate','path':str(droot/'run-007-collar-expanded-safe-replay/source-D-canonical14-stars-conditioned.npz'),'sha256':'3b3a5c183b49504aa00a4630cbc71b0bbecb0d5618dc8bbe958b5fd259971841'},
      {'asset_role':'run010_construction_map','path':str(run/'candidate-operational-target-map.json'),'sha256':'ec76d258264df284e8d8aa15d35d5bd53a2277a457e0576005ecff87f2bb2f4b'},
      {'asset_role':'run010_preflight_receipt','path':str(run/'candidate-preflight-receipt.json'),'sha256':'54c8ae6e3154c65e7253887e8ba99f37e22cad6c486ed0f7e9aa4e52c142cd0a'},
      {'asset_role':'run010_changed_face_exact_gates','path':str(run/'changed-exact-gates/summary.json'),'sha256':'f66f3525d81470e80a3c12aed1eb39743e3d6c38288e83fbd72c3b3fc36b13f0'},
      {'asset_role':'run010_exact_gate_producer','path':str(run/'run010_exact_changed_gates.py'),'sha256':'c5fb7b23f679db94b45da4715900a6d37d0a57e690dfc9c5bd56a004d8ef8113'},
    ]
    for a in assets:
        p=Path(a['path'])
        if not p.is_file() or sha(p)!=a['sha256']: raise AdapterError(f'{sid}: canonical14 parent proof pin failed: {p}')
    z=np.load(expected_source,allow_pickle=False)
    src_schema=json.loads(schema_text({k:np.asarray(z[k]) for k in z.files}))
    src_info=src_schema.get(str(sid))
    if not isinstance(src_info,dict): raise AdapterError(f'{sid}: source candidate does not carry this derived identity')
    for key in ('role','derived_operation','derived_parent_source_ids','face_count','identity_keys'):
        if src_info.get(key)!=info.get(key): raise AdapterError(f'{sid}: source candidate/schema field {key} differs from current pinned registry')
    out_ids=np.asarray(candidate['face_source_original_id'],dtype=np.int64);src_ids=np.asarray(z['face_source_original_id'],dtype=np.int64)
    out_rows=np.flatnonzero(out_ids==sid);src_rows=np.flatnonzero(src_ids==sid)
    if len(out_rows)!=13 or len(src_rows)!=13 or int(info.get('face_count',-1))!=13:
        raise AdapterError(f'{sid}: expected exactly 13 current/source identity rows')
    triple_keys=('face_source_face_id','face_source_identity0','face_source_identity1')
    out_triples=[tuple(int(np.asarray(candidate[k])[r]) for k in triple_keys) for r in out_rows]
    src_triples=[tuple(int(np.asarray(z[k])[r]) for k in triple_keys) for r in src_rows]
    if len(set(out_triples))!=13 or set(out_triples)!=set(src_triples):
        raise AdapterError(f'{sid}: source identity triples do not bijectively match the 13 output records')
    by_triple={t:r for t,r in zip(src_triples,src_rows)}
    outp=np.asarray(candidate['points_world_m'],dtype=np.float32);outf=np.asarray(candidate['faces'],dtype=np.int64)
    srcp=np.asarray(z['points_world_m'],dtype=np.float32);srcf=np.asarray(z['faces'],dtype=np.int64)
    for outrow,t in zip(out_rows,out_triples):
        srcrow=by_triple[t]
        if not np.array_equal(outp[outf[outrow]].view(np.uint32),srcp[srcf[srcrow]].view(np.uint32)):
            raise AdapterError(f'{sid}: directed F32 triangle differs from run010 source for identity {t}')
    info['sha256']=expected_source_sha
    info['identity_face_id_key']='face_source_face_id'
    info['identity_original_id_key']='face_source_original_id'
    info['identity_original_id']=sid
    info['derived_parent_assets']=assets
    info['source_geometry_scope']='13 exact source identity rows from pinned run010 source candidate; output directed F32 triangles match bitwise, complete operation ancestry remains in candidate CSR and run010 source.'
    return {'id':sid,'status':'PASS_SOURCE_IDENTITY_AND_PARENT_PROOF','path':str(expected_source),'sha256':expected_source_sha,
      'current_candidate_rows':out_rows.tolist(),'source_candidate_rows':src_rows.tolist(),
      'identity_triples':out_triples,'directed_F32_triangle_bits_equal':True,
      'parent_assets':assets,'parent_source_ids_preserved':info.get('derived_parent_source_ids'),
      'raw_schema_hash_literal_before_overlay':src_info.get('sha256'),
      'resolved_source_hash_from_external_run010_artifact_pin':expected_source_sha}

def semantic_parent_source_union(source_ids:np.ndarray,cache_rows:np.ndarray,producer_parent_values:np.ndarray)->list[int]:
    """Separate source-ID namespace from cache-row namespace using exact producer arrays."""
    ids=set(map(int,np.asarray(source_ids,dtype=np.int64).tolist()))
    rows=set(map(int,np.asarray(cache_rows,dtype=np.int64).tolist()))
    raw=set(map(int,np.asarray(producer_parent_values,dtype=np.int64).tolist()))
    if not ids or not rows: raise AdapterError('62000202 parent source/cache domains must both be nonempty')
    if ids & rows: raise AdapterError('62000202 source IDs collide with cache row IDs')
    if raw != ids | rows: raise AdapterError('62000202 original mixed parent list does not exactly equal typed source-ID union plus cache rows')
    return sorted(ids)

def verify_62000202_parent_lineage(candidate:dict[str,np.ndarray],schema:dict[str,Any],outdir:Path)->dict[str,Any]:
    """Reconstruct the true parent source-ID union and row crosswalk from pinned run003 ancestry."""
    sid=D202_ID; entry=schema.get(str(sid))
    if not isinstance(entry,dict) or entry.get('role')!='D' or entry.get('layer')!=2:
        raise AdapterError('62000202 must remain the registered D layer-2 derived source')
    source_path=Path(D202_LINEAGE_SOURCE); parent_path=Path(D202_PARENT_F32); report_path=Path(D202_LINEAGE_REPORT)
    for p,expected in ((source_path,D202_LINEAGE_SHA),(parent_path,D202_PARENT_F32_SHA),(report_path,D202_LINEAGE_REPORT_SHA)):
        if not p.is_file() or sha(p)!=expected: raise AdapterError(f'62000202 pinned lineage input hash failed: {p}')
    if Path(entry.get('path',''))!=source_path or entry.get('sha256')!=D202_LINEAGE_SHA:
        raise AdapterError('62000202 current registry source path/hash does not match pinned emitted lineage source')
    report=json.loads(report_path.read_text())
    if report.get('derived_source_sha256')!=D202_LINEAGE_SHA or report.get('parent_candidate_f32_sha256')!=D202_PARENT_F32_SHA:
        raise AdapterError('62000202 lineage report does not bind the pinned source and run003 parent')
    if (int(report.get('flip_input_zero_face_row',-1)),int(report.get('flip_neighbor_face_row',-1)))!=(144756,144762):
        raise AdapterError('62000202 local 2-2 parent face rows differ from pinned lineage report')
    expected_flip=np.asarray(report.get('flip_new_faces'),dtype=np.int64)
    if expected_flip.shape!=(2,3): raise AdapterError('62000202 flip output faces missing from report')
    with np.load(source_path,allow_pickle=False) as z: src={k:np.asarray(z[k]).copy() for k in z.files}
    with np.load(parent_path,allow_pickle=False) as z: par={k:np.asarray(z[k]).copy() for k in z.files}
    needed_src=('points_world_m','faces','source_candidate_face_rows','source_candidate_vertex_rows','face_ancestry_offsets','face_ancestry_input_cache_rows','face_ancestry_original_ids','face_ancestry_face_ids','face_ancestry_identity0','face_ancestry_identity1','face_ancestry_layer','face_ancestry_body','face_ancestry_bone','union_ancestry_offsets','union_ancestry_run_original_ids','union_ancestry_face_ids','union_ancestry_rows','parent_source_original_ids','parent_assets_json')
    needed_par=('points_world_m','faces','face_source_original_id','face_source_face_id','face_source_identity0','face_source_identity1','face_source_layer','face_source_body','face_source_bone','union_face_ancestry_offsets','union_face_ancestry_run_original_ids','union_face_ancestry_face_ids','union_face_ancestry_rows')
    for k in needed_src:
        if k not in src: raise AdapterError(f'62000202 source lineage lacks {k}')
    for k in needed_par:
        if k not in par: raise AdapterError(f'62000202 run003 parent lacks {k}')
    n=len(src['faces']); parent_rows=np.asarray(src['source_candidate_face_rows'],dtype=np.int64)
    if n!=29474 or parent_rows.shape!=(n,) or len(np.unique(parent_rows))!=n:
        raise AdapterError('62000202 output→run003 face-row map is not a 29,474-row bijection')
    vertex_rows=np.asarray(src['source_candidate_vertex_rows'],dtype=np.int64)
    if vertex_rows.shape!=(len(src['points_world_m']),) or len(np.unique(vertex_rows))!=len(vertex_rows):
        raise AdapterError('62000202 output→run003 vertex-row map is not bijective')
    if not np.array_equal(src['points_world_m'].astype(np.float32).view(np.uint32),par['points_world_m'][vertex_rows].astype(np.float32).view(np.uint32)):
        raise AdapterError('62000202 source view points differ from mapped run003 point bits')
    if np.any(parent_rows<0) or np.any(parent_rows>=len(par['faces'])): raise AdapterError('62000202 parent face row is out of range')
    flip_rows=np.flatnonzero(np.isin(parent_rows,[144756,144762]))
    if flip_rows.tolist()!=[n-2,n-1]: raise AdapterError('62000202 two local 2-2 children are not the final two source rows')
    mapped_faces=vertex_rows[np.asarray(src['faces'],dtype=np.int64)]
    direct=np.ones(n,dtype=bool);direct[flip_rows]=False
    direct_equal=np.all(mapped_faces[direct]==par['faces'][parent_rows[direct]],axis=1)
    if not np.all(direct_equal): raise AdapterError(f'62000202 has {int(np.count_nonzero(~direct_equal))} direct parent triangles that do not match')
    if not np.array_equal(mapped_faces[flip_rows],expected_flip): raise AdapterError('62000202 emitted 2-2 child triangles differ from pinned report')
    so=np.asarray(src['union_ancestry_offsets'],dtype=np.int64); po=np.asarray(par['union_face_ancestry_offsets'],dtype=np.int64)
    source_ids=np.asarray(src['union_ancestry_run_original_ids'],dtype=np.int64)
    source_fids=np.asarray(src['union_ancestry_face_ids'],dtype=np.int64)
    source_cache=np.asarray(src['union_ancestry_rows'],dtype=np.int64)
    if not np.array_equal(source_ids,np.asarray(src['face_ancestry_original_ids'],dtype=np.int64)) or not np.array_equal(source_fids,np.asarray(src['face_ancestry_face_ids'],dtype=np.int64)) or not np.array_equal(source_cache,np.asarray(src['face_ancestry_input_cache_rows'],dtype=np.int64)):
        raise AdapterError('62000202 union and per-face ancestry arrays disagree')
    if so.shape!=(n+1,) or po.ndim!=1: raise AdapterError('62000202 ancestry offsets malformed')
    true_ids=np.unique(source_ids)
    expected_ids=[62000100,62000106,62000107,62000108,100000061]
    if true_ids.tolist()!=expected_ids: raise AdapterError(f'62000202 typed run003 source-ID union differs: {true_ids.tolist()}')
    # Parent source IDs are independently recovered from the selected authenticated parent face rows.
    selected_parent_ids=np.unique(np.asarray(par['face_source_original_id'],dtype=np.int64)[parent_rows])
    if not np.array_equal(selected_parent_ids,true_ids): raise AdapterError('62000202 typed parent face rows yield a different source-ID union')
    raw_parent=np.asarray(src['parent_source_original_ids'],dtype=np.int64)
    corrected=semantic_parent_source_union(true_ids,source_cache,raw_parent)
    if entry.get('derived_parent_source_ids')!=raw_parent.tolist():
        raise AdapterError('62000202 registry raw parent list differs from immutable producer parent_source_original_ids')
    # All ten parent assets are immutable inputs named by both source record and registry.
    reg_assets=entry.get('derived_parent_assets')
    try: src_assets=json.loads(str(src['parent_assets_json'].ravel()[0]))
    except Exception as exc: raise AdapterError(f'62000202 source parent_assets_json is invalid: {exc}')
    if not isinstance(reg_assets,list) or len(reg_assets)!=10 or not isinstance(src_assets,list):
        raise AdapterError('62000202 parent asset registry/source list is incomplete')
    reg_keys={(a.get('path'),a.get('sha256')) for a in reg_assets}
    src_keys={(a.get('path'),a.get('sha256')) for a in src_assets}
    if reg_keys!=src_keys: raise AdapterError('62000202 registry and emitted-source parent asset pins differ')
    for path,expected in sorted(reg_keys):
        p=Path(path)
        if not p.is_file() or sha(p)!=expected: raise AdapterError(f'62000202 derived parent asset pin failed: {p}')
    for parent_id in corrected:
        pinfo=schema.get(str(parent_id))
        if not isinstance(pinfo,dict): raise AdapterError(f'62000202 true parent source {parent_id} is unregistered')
        if int(parent_id)==100000061 and pinfo.get('role')!='BONE': raise AdapterError('62000202 BONE parent role is not preserved')
        if int(parent_id)!=100000061 and pinfo.get('role')!='D': raise AdapterError(f'62000202 D parent {parent_id} has unexpected registry role')
    # Build parent face-row reverse map over the selected run003 faces; every ancestry entry must map uniquely.
    key_to_parent={}
    for r in parent_rows.tolist():
        r=int(r)
        for j in range(int(po[r]),int(po[r+1])):
            key=(int(par['union_face_ancestry_run_original_ids'][j]),int(par['union_face_ancestry_face_ids'][j]),int(par['union_face_ancestry_rows'][j]))
            key_to_parent.setdefault(key,[]).append(r)
    out_rows=[];parent_face_rows=[];cache_rows=[];xids=[];xfids=[];xi0=[];xi1=[];xl=[];xb=[];xbon=[]
    parent_typed_names=('original_id','face_id','identity0','identity1','layer','body','bone')
    source_typed_names=('original_ids','face_ids','identity0','identity1','layer','body','bone')
    nentries=0; direct_ancestry_rows=0; merged_ancestry_rows=0
    for i,r0 in enumerate(parent_rows):
        r=int(r0); child=[]
        for j in range(int(so[i]),int(so[i+1])):
            key=(int(source_ids[j]),int(source_fids[j]),int(source_cache[j]))
            matches=key_to_parent.get(key,[])
            if len(matches)!=1: raise AdapterError(f'62000202 ancestry {key} maps to {len(matches)} selected run003 parent face rows')
            pr=matches[0]
            if pr not in ([144756,144762] if r in (144756,144762) else [r]):
                raise AdapterError(f'62000202 ancestry mapped to unrelated parent face row {pr} for output row {i}')
            for sk,pk in zip(source_typed_names,parent_typed_names):
                sv=int(np.asarray(src['face_ancestry_'+sk])[j]);pv=int(np.asarray(par['face_source_'+pk])[pr])
                if sv!=pv: raise AdapterError(f'62000202 ancestry typed field {sk} differs from run003 parent face {pr}')
            child.append(pr)
            out_rows.append(i);parent_face_rows.append(pr);cache_rows.append(int(source_cache[j]));xids.append(int(source_ids[j]));xfids.append(int(source_fids[j]))
            xi0.append(int(src['face_ancestry_identity0'][j]));xi1.append(int(src['face_ancestry_identity1'][j]));xl.append(int(src['face_ancestry_layer'][j]));xb.append(int(src['face_ancestry_body'][j]));xbon.append(int(src['face_ancestry_bone'][j]))
            nentries+=1
        expected_parents=[r] if r not in (144756,144762) else [144756,144762]
        if sorted(set(child))!=sorted(expected_parents): raise AdapterError(f'62000202 output row {i} ancestry parent crosswalk incomplete')
        if r in (144756,144762): merged_ancestry_rows+=1
        else: direct_ancestry_rows+=1
    crosswalk={
      'output_face_rows':np.asarray(out_rows,dtype=np.int64),
      'parent_run003_candidate_face_rows':np.asarray(parent_face_rows,dtype=np.int64),
      'parent_input_cache_face_rows':np.asarray(cache_rows,dtype=np.int64),
      'source_original_ids':np.asarray(xids,dtype=np.int64),
      'source_face_ids':np.asarray(xfids,dtype=np.int64),
      'source_identity0':np.asarray(xi0,dtype=np.int64),
      'source_identity1':np.asarray(xi1,dtype=np.int64),
      'source_layer':np.asarray(xl,dtype=np.int64),
      'source_body':np.asarray(xb,dtype=np.int64),
      'source_bone':np.asarray(xbon,dtype=np.int64),
      'producer_parent_source_original_ids_mixed_namespace':raw_parent.copy(),
      'true_derived_parent_source_ids':np.asarray(corrected,dtype=np.int64),
    }
    if nentries!=29476: raise AdapterError(f'62000202 ancestry crosswalk has {nentries} entries, expected 29,476')
    xpath=outdir/'62000202-run003-parent-face-row-crosswalk.npz';np.savez_compressed(xpath,**crosswalk); xsha=sha(xpath)
    active_primary=int(np.count_nonzero(np.asarray(candidate['face_source_original_id'])==sid))
    active_csr=int(np.count_nonzero(np.asarray(candidate['quality_star_patch_union_face_source_original_id'])==sid))
    proof={'id':sid,'status':'PASS_AUTHENTICATED_PARENT_FACE_AND_SOURCE_ID_UNION',
      'source_path':str(source_path),'source_sha256':D202_LINEAGE_SHA,'parent_run003_path':str(parent_path),'parent_run003_sha256':D202_PARENT_F32_SHA,
      'lineage_report_path':str(report_path),'lineage_report_sha256':D202_LINEAGE_REPORT_SHA,
      'source_faces':int(n),'source_vertex_rows':int(len(vertex_rows)),'output_to_parent_face_rows_bijective':True,
      'output_to_parent_point_coordinates_bitwise_equal':True,'direct_face_rows_exactly_equal':int(np.count_nonzero(direct_equal)),
      '2_to_2_flip_child_rows':flip_rows.tolist(),'2_to_2_flip_parent_rows':[144756,144762],
      'flip_child_faces_global':mapped_faces[flip_rows].tolist(),'parent_typed_ancestry_rows_verified':int(nentries),
      'direct_rows_with_exact_one_parent_face':int(direct_ancestry_rows),'flip_rows_with_complete_two_parent_union':int(merged_ancestry_rows),
      'true_parent_source_ids':corrected,'true_parent_source_id_count':len(corrected),
      'source_ids_from_selected_run003_typed_faces_equal':True,'registered_parent_roles':{str(i):schema[str(i)].get('role') for i in corrected},
      'original_producer_parent_source_original_ids_count':int(len(raw_parent)),'producer_parent_list_unique_count':int(len(np.unique(raw_parent))),
      'producer_parent_list_semantics':'mixed namespace: exact union of 29,474 parent input/cache face row values and five true source IDs; not valid as source IDs',
      'corrected_derived_parent_source_ids':corrected,'original_producer_parent_source_original_ids_preserved':raw_parent.tolist(),
      'parent_assets_all_hash_verified':len(reg_assets),'crosswalk_path':str(xpath),'crosswalk_sha256':xsha,
      'current_candidate_primary_rows_for_id':active_primary,'current_candidate_CSR_rows_for_id':active_csr,
      'current_geometry_modified':False,'source_claim':'complete patch-level typed union ancestry only; no per-child geometric overlap claim'}
    proof_path=outdir/'62000202-parent-lineage-proof.json';proof_path.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
    proof['proof_path']=str(proof_path);proof['proof_sha256']=sha(proof_path)
    return proof

def verify_inactive_temporary_cap(candidate:dict[str,np.ndarray],schema:dict[str,Any],outdir:Path)->dict[str,Any]:
    """Authenticate inactive 41-face temporary closure and keep it out of anatomy/source projection."""
    sid=80000185;info=schema.get(str(sid))
    if not isinstance(info,dict) or info.get('role')!='D' or info.get('derived_operation')!='temporary_head_cap_for_closed_boolean_input':
        raise AdapterError('80000185 is not the registered temporary run037 head cap')
    if Path(info.get('path',''))!=Path(TEMP_CAP_SOURCE) or sha(Path(TEMP_CAP_SOURCE))!=TEMP_CAP_SOURCE_SHA:
        raise AdapterError('80000185 temporary cap source pin failed')
    primary=np.asarray(candidate['face_source_original_id'],dtype=np.int64);csr=np.asarray(candidate['quality_star_patch_union_face_source_original_id'],dtype=np.int64)
    if np.any(primary==sid) or np.any(csr==sid): raise AdapterError('80000185 temporary cap unexpectedly has active current candidate tuples')
    for p,expected in ((Path(TEMP_CAP_GEOM),TEMP_CAP_GEOM_SHA),(Path(TEMP_CAP_RECEIPT),TEMP_CAP_RECEIPT_SHA),(Path(TEMP_CAP_PARENT),TEMP_CAP_PARENT_SHA),(Path(TEMP_CAP_PRODUCER),TEMP_CAP_PRODUCER_SHA),(Path(TEMP_CAP_PARENT_PRODUCER),TEMP_CAP_PARENT_PRODUCER_SHA)):
        if not p.is_file() or sha(p)!=expected: raise AdapterError(f'80000185 cap construction ancestry pin failed: {p}')
    rec=json.loads(Path(TEMP_CAP_RECEIPT).read_text())
    if rec.get('candidate_sha256')!='f94b61ca212f143a4b7257fb2bae6adcb3898e7642d29698e173039458c7cc55' or rec.get('closed_parent_sha256')!=TEMP_CAP_PARENT_SHA or rec.get('output_sha256')!=TEMP_CAP_GEOM_SHA:
        raise AdapterError('80000185 cap receipt does not bind the run037 open candidate/closed parent/output')
    rows=np.asarray(rec.get('cap_parent_face_rows'),dtype=np.int64)
    if rows.shape!=(41,) or len(np.unique(rows))!=41 or int(rec.get('cap_faces',-1))!=41:
        raise AdapterError('80000185 cap parent face-row list is not 41 unique rows')
    with np.load(TEMP_CAP_PARENT,allow_pickle=False) as z: par={k:np.asarray(z[k]).copy() for k in z.files}
    with np.load(TEMP_CAP_GEOM,allow_pickle=False) as z: geom={k:np.asarray(z[k]).copy() for k in z.files}
    with np.load(TEMP_CAP_SOURCE,allow_pickle=False) as z: closed={k:np.asarray(z[k]).copy() for k in z.files}
    cap=np.asarray(geom['closed_cap_faces'],dtype=np.int64)
    if not np.array_equal(cap,par['faces_f64'][rows]): raise AdapterError('80000185 closed cap triangles do not equal pinned run036 parent faces')
    if not np.array_equal(cap,closed['cap_faces_from_run036_closed_parent_rows']): raise AdapterError('80000185 diagnostic source cap rows differ from run037 cap geometry')
    if not np.array_equal(geom['points_world_m'].view(np.uint32),closed['points_world_m_f64'].astype(np.float32).view(np.uint32)):
        raise AdapterError('80000185 cap vertex point mapping differs from pinned run037 F32 output')
    if len(geom['faces'])!=136013 or len(cap)!=41 or int(info.get('face_count',-1))!=41:
        raise AdapterError('80000185 cap/open face count differs from source receipt')
    cap_path=outdir/'80000185-temporary-cap-parent-face-crosswalk.npz'
    np.savez_compressed(cap_path,temporary_cap_local_face_rows=np.arange(41,dtype=np.int64),run036_closed_parent_face_rows=rows,
      run036_parent_triangle_point_rows=par['faces_f64'][rows],run037_open_face_count=np.asarray([136013],dtype=np.int64),
      cap_face_count=np.asarray([41],dtype=np.int64))
    return {'id':sid,'status':'PASS_INACTIVE_TEMPORARY_CONSTRUCTION_CAP','active_primary_tuple_rows':0,'active_CSR_tuple_rows':0,
      'source_path':TEMP_CAP_SOURCE,'source_sha256':TEMP_CAP_SOURCE_SHA,'geometry_receipt_path':TEMP_CAP_RECEIPT,'geometry_receipt_sha256':TEMP_CAP_RECEIPT_SHA,
      'run037_geometry_path':TEMP_CAP_GEOM,'run037_geometry_sha256':TEMP_CAP_GEOM_SHA,'run036_closed_parent_path':TEMP_CAP_PARENT,'run036_closed_parent_sha256':TEMP_CAP_PARENT_SHA,
      'parent_face_rows':rows.tolist(),'parent_face_count':len(rows),'run037_open_face_count':136013,'temporary_cap_face_count':41,
      'cap_triangles_exact_parent_rows':True,'cap_f32_point_domain_exact':True,
      'interpretation':'temporary closure used to create a closed Boolean input for a diagnostic hull/occupancy process; not an anatomical target patch and not present in current primary/CSR tuple rows',
      'active_authoring_projection':False,'crosswalk_path':str(cap_path),'crosswalk_sha256':sha(cap_path),
      'producer_script':TEMP_CAP_PRODUCER,'producer_script_sha256':TEMP_CAP_PRODUCER_SHA,
      'parent_producer_script':TEMP_CAP_PARENT_PRODUCER,'parent_producer_script_sha256':TEMP_CAP_PARENT_PRODUCER_SHA}

def project_layers(candidate:dict[str,np.ndarray],schema:dict[str,Any])->tuple[dict[str,np.ndarray],dict[str,Any]]:
    out={k:np.asarray(v).copy() for k,v in candidate.items()}
    selected_registry=projection_ids(schema)
    if 'face_source_original_id' not in out or 'face_source_layer' not in out:
        raise AdapterError('missing primary source tuple arrays')
    primary_id=out['face_source_original_id']; primary_layer=out['face_source_layer']
    if primary_id.shape!=primary_layer.shape: raise AdapterError('primary source tuple extent mismatch')
    c_id_key='quality_star_patch_union_face_source_original_id';c_layer_key='quality_star_patch_union_face_source_layer'
    if c_id_key not in out or c_layer_key not in out: raise AdapterError('missing typed union source tuples')
    csr_id=out[c_id_key];csr_layer=out[c_layer_key]
    if csr_id.shape!=csr_layer.shape: raise AdapterError('CSR source tuple extent mismatch')
    used_ids=set(map(int,np.unique(primary_id)))|set(map(int,np.unique(csr_id)))
    selected=[sid for sid in selected_registry if sid in used_ids]
    inactive_registry_ids=[sid for sid in selected_registry if sid not in used_ids]
    expected_by_id={}
    for sid in selected:
        expected_by_id[sid]=_expected_source_codes(sid,schema[str(sid)],schema,
          primary_id.astype(np.int64),primary_layer.astype(np.int64),csr_id.astype(np.int64),csr_layer.astype(np.int64))
        pr=np.flatnonzero(primary_id==sid);cr=np.flatnonzero(csr_id==sid)
        observed=set(map(int,primary_layer[pr].tolist()))|set(map(int,csr_layer[cr].tolist()))
        if not observed: continue
        if not observed.issubset(expected_by_id[sid]):
            raise AdapterError(f'{sid}: primary/CSR source tuples {sorted(observed)} disagree with authenticated source-layer set {sorted(expected_by_id[sid])}')
    selected_arr=np.asarray(selected,dtype=np.int64)
    primary_mask=np.isin(primary_id,selected_arr);primary_changed_mask=primary_mask & (primary_layer!=2)
    primary_rows=np.flatnonzero(primary_changed_mask).astype(np.int64)
    primary_selected_rows=np.flatnonzero(primary_mask).astype(np.int64)
    primary_from={str(int(code)):int(np.count_nonzero(primary_layer[primary_mask]==code)) for code in np.unique(primary_layer[primary_mask])}
    projected_primary_ids=sorted(map(int,np.unique(primary_id[primary_changed_mask])))
    primary_layer[primary_mask]=2
    csr_mask=np.isin(csr_id,selected_arr);csr_changed_mask=csr_mask & (csr_layer!=2)
    csr_rows=np.flatnonzero(csr_changed_mask).astype(np.int64);csr_selected_rows=np.flatnonzero(csr_mask).astype(np.int64)
    csr_from={str(int(code)):int(np.count_nonzero(csr_layer[csr_mask]==code)) for code in np.unique(csr_layer[csr_mask])}
    projected_csr_ids=sorted(map(int,np.unique(csr_id[csr_changed_mask])))
    csr_layer[csr_mask]=2
    # Keep non-D roles, non-derived rows, and DERIVED_PATCH records unchanged.
    primary_derived_patch_ids=sorted(int(i) for i in np.unique(primary_id)
        if schema.get(str(int(i)),{}).get('role')=='DERIVED_PATCH')
    csr_derived_patch_ids=sorted(int(i) for i in np.unique(csr_id)
        if schema.get(str(int(i)),{}).get('role')=='DERIVED_PATCH')
    per_source=[]
    for sid in selected:
        pr=np.flatnonzero(primary_id==sid);cr=np.flatnonzero(csr_id==sid)
        observed=set(map(int,candidate['face_source_layer'][pr].tolist()))|set(map(int,candidate[c_layer_key][cr].tolist()))
        per_source.append({'id':sid,'role':schema[str(sid)].get('role'),
          'source_schema_layer_code':schema[str(sid)].get('layer'),
          'observed_source_layer_codes':sorted(observed),
          'authenticated_source_layer_codes':sorted(expected_by_id[sid]),
          'authoring_layer_code':2,'primary_rows':int(len(pr)),'csr_rows':int(len(cr))})
    report={'rule':'Every used registry role-D source with a nonempty derived_operation maps to authoring tuple layer 2; original source tuple codes and complete parent unions remain byte-for-byte in provenance.',
      'source_registry_projection_ids':selected,'projection_id_count':len(selected),
      'inactive_registry_derived_D_ids_without_candidate_tuple_rows':inactive_registry_ids,
      'primary_selected_rows':primary_selected_rows.tolist(),'primary_selected_count':int(len(primary_selected_rows)),
      'primary_changed_rows':primary_rows.tolist(),'primary_changed_count':int(len(primary_rows)),
      'primary_changed_source_ids':projected_primary_ids,'primary_rows_by_original_source_layer':primary_from,
      'csr_selected_rows':csr_selected_rows.tolist(),'csr_selected_count':int(len(csr_selected_rows)),
      'csr_changed_rows':csr_rows.tolist(),'csr_changed_count':int(len(csr_rows)),
      'csr_changed_source_ids':projected_csr_ids,'csr_rows_by_original_source_layer':csr_from,
      'per_source_projection':per_source,
      'primary_derived_patch_ids_left_unchanged':primary_derived_patch_ids,
      'csr_derived_patch_ids_left_unchanged':csr_derived_patch_ids,
      'BONE_layer3_primary_rows_unchanged':int(np.count_nonzero((primary_layer==3)&np.isin(primary_id,[int(k) for k,v in schema.items() if v.get('role')=='BONE']))),
      'BONE_layer3_CSR_rows_unchanged':int(np.count_nonzero((csr_layer==3)&np.isin(csr_id,[int(k) for k,v in schema.items() if v.get('role')=='BONE'])))}
    return out,report

def load590(helper_path:Path):
    if sha(helper_path)!=PIN_590_HELPER_SHA256: raise AdapterError('590 helper source hash mismatch')
    spec=importlib.util.spec_from_file_location('registered_authoring_view_590_pinned',helper_path)
    if spec is None or spec.loader is None: raise AdapterError('cannot load pinned 590 helper')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def verify_62000204(candidate:dict[str,np.ndarray],registry:dict[str,Any],filtered_path:Path,
                    registry_source:dict[str,np.ndarray])->dict[str,Any]:
    info=registry.get(str(DERIVED_ID))
    if not isinstance(info,dict) or info.get('role')!='D' or not info.get('derived_operation'):
        raise AdapterError('62000204 is not pinned as a derived D source')
    if int(info.get('face_count',-1))!=15 or info.get('raw_geometry_sha256')!='16a5d942467ce894c111caaf23292f553451bf0d8a6f2e92b0308b574b48879a':
        raise AdapterError('62000204 source face count/raw geometry pin differs')
    if info.get('derived_parent_source_ids')!=[5,62000202]:
        raise AdapterError('62000204 original parent source IDs changed')
    if not info.get('derived_parent_assets'):
        raise AdapterError('62000204 original parent asset pins missing')
    z=np.load(filtered_path,allow_pickle=False)
    ids=np.asarray(candidate['face_source_original_id'],dtype=np.int64)
    ident0=np.asarray(candidate['face_source_identity0'],dtype=np.int64)
    ident1=np.asarray(candidate['face_source_identity1'],dtype=np.int64)
    faceid=np.asarray(candidate['face_source_face_id'],dtype=np.int64)
    rows=np.flatnonzero(ids==DERIVED_ID)
    if len(rows)!=15: raise AdapterError(f'62000204 candidate face count changed: {len(rows)}')
    order=np.argsort(ident0[rows],kind='stable');rows=rows[order]
    if not np.array_equal(ident0[rows],np.arange(15,dtype=np.int64)): raise AdapterError('62000204 identity0 is not exactly 0..14')
    if not np.all(ident1[rows]==-1): raise AdapterError('62000204 identity1 is not exactly -1')
    if not np.all(faceid[rows]==-1): raise AdapterError('62000204 native face IDs must remain the source sentinel -1')
    if np.any(candidate['face_source_layer'][rows]!=2): raise AdapterError('62000204 layer tuples unexpectedly changed')
    sids=np.asarray(z['face_source_original_id'],dtype=np.int64)
    sr0=np.asarray(z['face_source_identity0'],dtype=np.int64)
    sr1=np.asarray(z['face_source_identity1'],dtype=np.int64)
    sfid=np.asarray(z['face_source_face_id'],dtype=np.int64)
    sr=np.flatnonzero(sids==DERIVED_ID)
    sr=sr[np.argsort(sr0[sr],kind='stable')]
    if len(sr)!=15 or not np.array_equal(sr0[sr],np.arange(15,dtype=np.int64)) or not np.all(sr1[sr]==-1) or not np.all(sfid[sr]==-1):
        raise AdapterError('filtered source-view identity tuples do not match current 62000204 tuples')
    cp=np.asarray(candidate['points_world_m'],dtype=np.float32);cf=np.asarray(candidate['faces'],dtype=np.int64)
    sp=np.asarray(z['points_world_m'],dtype=np.float32);sf=np.asarray(z['faces'],dtype=np.int64)
    for local,(cr,sr0row) in enumerate(zip(rows,sr)):
        if not np.array_equal(cp[cf[cr]].view(np.uint32),sp[sf[sr0row]].view(np.uint32)):
            raise AdapterError(f'62000204 directed F32 triangle differs from filtered D128 source at face identity {local}')
    raw=str(np.asarray(z['raw_geometry_sha256']).ravel()[0])
    if raw!=info['raw_geometry_sha256']: raise AdapterError('filtered source view raw geometry hash mismatch')
    registry128=json.loads(schema_text(registry_source))
    if str(DERIVED_ID) not in registry128: raise AdapterError('D128 pinned registry lacks 62000204')
    if registry128[str(DERIVED_ID)].get('raw_geometry_sha256')!=info['raw_geometry_sha256']:
        raise AdapterError('592 candidate registry raw geometry differs from D128 registry')
    return {'id':DERIVED_ID,'face_count':15,'candidate_rows':rows.tolist(),'source_view_rows':sr.tolist(),
      'face_source_face_id_preserved':True,'identity0':list(range(15)),'identity1':[-1]*15,
      'directed_f32_triangle_coordinate_bits_equal':True,'raw_geometry_sha256':raw,
      'derived_parent_source_ids':info['derived_parent_source_ids'],
      'derived_parent_assets':info['derived_parent_assets'],
      'complete_patch_union_claim':'retained from candidate face_ancestry and quality_star_patch_union CSR arrays; no per-child geometric overlap claim'}

def _json_array(text:str)->np.ndarray:
    return np.asarray([text],dtype=f'<U{max(1,len(text))}')

def build(args:argparse.Namespace)->dict[str,Any]:
    paths={k:Path(getattr(args,k)) for k in ('candidate','registry_source','filtered_source','helper590','exporter230')}
    expected={'candidate':INPUT_SHA256,'registry_source':REGISTRY_SHA256,'filtered_source':FILTERED_SOURCE_SHA256,
      'helper590':PIN_590_HELPER_SHA256,'exporter230':PIN_230_EXPORTER_SHA256}
    # v3 required this historical ring1 artifact as a hash-only input although
    # no adapter, source-view, or existing230 path consumes it. Keep the pin and
    # recovery state in the receipt; never substitute different bytes.
    ring1_audit={'expected_path':PIN_RING1_EXPECTED_PATH,'expected_sha256':RING1_SHA256,
      'used_by_v4_adapter':False,'used_by_existing230_D_validation':False,
      'status':'NOT_FOUND_UNUSED_HISTORICAL_AUXILIARY'}
    if args.ring1_source:
        rp=Path(args.ring1_source)
        if not rp.is_file(): ring1_audit.update(status='NOT_FOUND_UNUSED_HISTORICAL_AUXILIARY',provided_path=str(rp))
        else:
            actual=sha(rp)
            if actual!=RING1_SHA256: raise AdapterError(f'provided historical ring1 source hash mismatch: {actual} != {RING1_SHA256}')
            ring1_audit.update(status='PIN_VERIFIED_UNUSED_HISTORICAL_AUXILIARY',provided_path=str(rp))
    for name,path in paths.items():
        if not path.is_file(): raise AdapterError(f'{name} file unavailable: {path}')
        actual=sha(path)
        if actual!=expected[name]: raise AdapterError(f'{name} pinned SHA mismatch: {actual} != {expected[name]}')
    for path,pin in (PIN_CHILD_READER,PIN_ROLE_CONSUMER):
        p=Path(path)
        if not p.is_file() or sha(p)!=pin: raise AdapterError(f'native role/source consumer pin mismatch: {p}')
    outdir=Path(args.out_dir)
    if outdir.exists(): raise AdapterError(f'refuse existing output directory {outdir}')
    outdir.mkdir(parents=True)
    loaded=np.load(paths['candidate'],allow_pickle=False)
    candidate={k:np.asarray(loaded[k]).copy() for k in loaded.files}
    reg_z=np.load(paths['registry_source'],allow_pickle=False)
    candidate_schema=json.loads(schema_text(candidate));registry_schema=json.loads(schema_text({k:np.asarray(v) for k,v in reg_z.items()}))
    # The existing 592 D schema must be a byte-equivalent semantic copy of the pinned D128 registry.
    if candidate_schema!=registry_schema: raise AdapterError('592 D source registry differs from pinned D128 registry')
    parent_path=str(np.asarray(candidate.get('native_authoring_source_schema_parent_path',[])).ravel()[0])
    parent_sha=str(np.asarray(candidate.get('native_authoring_source_schema_parent_sha256',[])).ravel()[0])
    if parent_path!=str(paths['registry_source']) or parent_sha!=REGISTRY_SHA256:
        raise AdapterError('592 D authoring view parent path/SHA does not match pinned D128 registry')
    idproof=verify_62000204(candidate,candidate_schema,paths['filtered_source'],{k:np.asarray(v) for k,v in reg_z.items()})
    lineage202proof=verify_62000202_parent_lineage(candidate,candidate_schema,outdir)
    cap800proof=verify_inactive_temporary_cap(candidate,candidate_schema,outdir)
    census_path=Path(CENSUS_PATH)
    if not census_path.is_file() or sha(census_path)!=CENSUS_SHA: raise AdapterError('aggregate current primary+CSR metadata census pin failed')
    census=json.loads(census_path.read_text())
    if census.get('candidate_sha256')!=INPUT_SHA256 or census.get('active_id_count')!=470 or census.get('primary_tuple_rows')!=215335 or census.get('csr_tuple_rows')!=662451:
        raise AdapterError('aggregate source-metadata census does not bind the current candidate counts')
    layer3proof=verify_layer3_provenance(candidate,candidate_schema)
    # Build an immutable derivative source view with explicit aliases for existing230's identity filter.
    sv=np.load(paths['filtered_source'],allow_pickle=False)
    source_view={k:np.asarray(sv[k]).copy() for k in sv.files}
    source_view['topology_patch_output_face_row']=np.asarray(source_view['face_source_identity0'],dtype=np.int64).copy()
    source_view['unused']=np.asarray(source_view['face_source_identity1'],dtype=np.int64).copy()
    source_view_path=outdir/'D-derived-source-62000204-filtered-D128-existing230.npz'
    np.savez_compressed(source_view_path,**source_view)
    source_view_sha=sha(source_view_path)
    overlaid=copy.deepcopy(candidate_schema)
    entry202=overlaid[str(D202_ID)]
    original_entry202=copy.deepcopy(entry202)
    entry202['producer_original_derived_parent_source_ids']=entry202['derived_parent_source_ids']
    entry202['derived_parent_source_ids']=lineage202proof['corrected_derived_parent_source_ids']
    entry202['derived_parent_source_ids_semantics']='authenticated unique source original-ID union from pinned run003 selected face rows and complete typed ancestry; producer list retained separately was a mixed cache-row/source-ID namespace'
    entry202['derived_parent_face_crosswalk_path']=lineage202proof['crosswalk_path']
    entry202['derived_parent_face_crosswalk_sha256']=lineage202proof['crosswalk_sha256']
    entry202['derived_parent_lineage_proof_path']=lineage202proof['proof_path']
    entry202['derived_parent_lineage_proof_sha256']=lineage202proof['proof_sha256']
    entry=overlaid[str(DERIVED_ID)]
    original_entry=copy.deepcopy(entry)
    if entry.get('identity_keys')!=['topology_patch_output_face_row','unused']:
        raise AdapterError('62000204 original identity key contract changed')
    entry.update(path=str(source_view_path),sha256=source_view_sha,
      identity_face_id_key='face_source_face_id',
      identity_original_id_key='face_source_original_id',identity_original_id=DERIVED_ID,
      source_geometry_scope='exact D128-filtered 15-face source view; parent patch union retained separately')
    canonical14proof=verify_100000186(candidate,overlaid)
    projected,layer_report=project_layers(candidate,overlaid)
    if not layer_report['source_registry_projection_ids']: raise AdapterError('pinned D128 registry contains no D-derived layer rule')
    # Keep the source registry's layer field untouched. These additional fields
    # explicitly identify the separate authoring namespace used by existing230.
    for sid in layer_report['source_registry_projection_ids']:
        entry=overlaid[str(sid)]
        if entry.get('role')!='D' or not entry.get('derived_operation'):
            raise AdapterError(f'{sid}: authoring projection escaped a derived D source')
        entry['authoring_role_layer']=2
        entry['authoring_layer_projection']={
          'source_schema_layer_code':candidate_schema[str(sid)].get('layer'),
          'observed_source_layer_codes':next(p['observed_source_layer_codes'] for p in layer_report['per_source_projection'] if p['id']==sid),
          'authoring_tuple_layer_code':2,
          'rule':'role-directed derived D authoring projection; original source tuples are retained in the provenance archive'}
    # Reassert that special 62000204 primary and CSR tuples, including face_id=-1, are byte-identical.
    def id_rows(a,key,sid): return np.flatnonzero(np.asarray(a[key])==sid)
    for prefix,idkey in [('face_source_','face_source_original_id'),('quality_star_patch_union_face_source_','quality_star_patch_union_face_source_original_id')]:
        source_id=np.asarray(candidate[idkey]);dest_id=np.asarray(projected[idkey]); rows=np.flatnonzero(source_id==DERIVED_ID)
        for field in TUPLE:
            k=prefix+field
            if k in candidate and not np.array_equal(candidate[k][rows],projected[k][rows]):
                raise AdapterError(f'62000204 typed tuple unexpectedly changed: {k}')
    schema_json=json.dumps(overlaid,sort_keys=True,separators=(',',':'))
    schema_source={'face_source_identity_schema_by_original_id_json':_json_array(schema_json)}
    helper=load590(paths['helper590'])
    prepared,counts=helper.prepare(projected,schema_source)
    # Keep exact parent-registry locator in this authoring namespace.
    prepared['native_authoring_source_schema_parent_path']=np.asarray([str(paths['registry_source'])],dtype=f'<U{len(str(paths["registry_source"]))}')
    prepared['native_authoring_source_schema_parent_sha256']=np.asarray([REGISTRY_SHA256],dtype='<U64')
    prepared['native_authoring_schema_adapter']=np.asarray(['registered-authoring-view-590-D-593.v5'])
    candidate_path=outdir/'candidate-D-590-derived-authoring.npz'
    np.savez_compressed(candidate_path,**prepared)
    # Snapshot every original primary and CSR tuple row plus exact parent face ancestry.
    prov={}
    for field in TUPLE:
        prov['original_primary_'+field]=np.asarray(candidate['face_source_'+field]).copy()
        prov['original_csr_'+field]=np.asarray(candidate['quality_star_patch_union_face_source_'+field]).copy()
    prov['original_primary_face_rows']=np.arange(len(candidate['faces']),dtype=np.int64)
    prov['original_quality_star_patch_union_offsets']=np.asarray(candidate['quality_star_patch_union_offsets']).copy()
    for k in ('face_ancestry_offsets','face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows'):
        if k in candidate: prov['original_'+k]=np.asarray(candidate[k]).copy()
    prov['62000202_original_producer_parent_source_original_ids_mixed_namespace']=np.asarray(lineage202proof['producer_parent_source_original_ids'],dtype=np.int64)
    prov['62000202_corrected_parent_source_id_union']=np.asarray(lineage202proof['corrected_derived_parent_source_ids'],dtype=np.int64)
    prov['primary_projected_layer_rows']=np.asarray(layer_report['primary_changed_rows'],dtype=np.int64)
    prov['csr_projected_layer_rows']=np.asarray(layer_report['csr_changed_rows'],dtype=np.int64)
    prov['projected_derived_source_ids']=np.asarray(layer_report['source_registry_projection_ids'],dtype=np.int64)
    provenance_path=outdir/'original-primary-csr-tuples-and-patch-ancestry.npz'
    np.savez_compressed(provenance_path,**prov)
    source_registry_path=outdir/'D128-original-source-registry.json'
    source_registry_path.write_text(json.dumps(candidate_schema,indent=2,sort_keys=True)+'\n')
    special_entry=overlaid[str(DERIVED_ID)]
    overlay={'schema':'registered-authoring-D-source-schema-overlay.v5','source_registry_path':str(paths['registry_source']),
      'source_registry_sha256':REGISTRY_SHA256,'original_source_entry':original_entry,
      'authoring_entry':special_entry,'filtered_source_view_path':str(source_view_path),'filtered_source_view_sha256':source_view_sha,
      'source_identity_rule':'62000204 face_source_face_id=-1 is retained and matched against the filtered D128 view; identity0 is exactly 0..14, identity1=-1; no face-ID invention.',
      'derived_layer_projection':layer_report,'layer3_parent_provenance_audit':layer3proof,
      '62000202_original_source_entry':original_entry202,'62000202_authoring_source_entry':overlaid[str(D202_ID)],
      '62000202_parent_lineage_semantic_audit':lineage202proof,'80000185_inactive_temporary_cap_audit':cap800proof,
      'aggregate_source_metadata_census':{'path':CENSUS_PATH,'sha256':CENSUS_SHA,'schema_ids':int(census['schema_id_count']),'active_ids':int(census['active_id_count']),'primary_active_ids':int(census['active_primary_id_count']),'CSR_active_ids':int(census['active_csr_id_count']),'primary_tuple_rows':int(census['primary_tuple_rows']),'CSR_tuple_rows':int(census['csr_tuple_rows']),'invalid_category_counts_before_authoring_overlays':census['invalid_category_counts']},
      'unlayered_derived_provenance_audit':audit_unlayered_derived_provenance(candidate,candidate_schema),
      'canonical14_source_adapter':canonical14proof,
      'derived_D_authoring_projection':'source tuple codes (including 3 for BONE parents) remain in the original tuple archive; derived role-D output rows are projected to authoring layer 2 in a separate authoring view.',
      'source_to_authoring_layer_map':[{'id':sid,'role':'D','source_schema_layer_code':candidate_schema[str(sid)].get('layer'),
          'observed_source_layer_codes':next(p['observed_source_layer_codes'] for p in layer_report['per_source_projection'] if p['id']==sid),'authoring_layer_code':2} for sid in layer_report['source_registry_projection_ids']],
      'DERIVED_PATCH_layer_tuples_modified':False,
      'raw_geometry_sha256':special_entry['raw_geometry_sha256'],'physical_candidate_modified':False}
    overlay_path=outdir/'authoring-schema-overlay.json'
    overlay_path.write_text(json.dumps(overlay,indent=2,sort_keys=True)+'\n')
    # Call the pinned actual existing230 source validator over the emitted file.
    source_dir=paths['exporter230'].parent;sys.path.insert(0,str(source_dir))
    spec=importlib.util.spec_from_file_location('existing230_e962',paths['exporter230'])
    if spec is None or spec.loader is None: raise AdapterError('could not load existing230 exporter')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    validation={'function':'validate_candidate(path, layer, require_f32=True)','source_commit':'e962da30203adf7a83bf751f5fa169f439006f56',
      'source_path':str(paths['exporter230']),'source_sha256':sha(paths['exporter230']),'status':'PENDING'}
    try:
        v=mod.validate_candidate(candidate_path,'D',require_f32=True)
        validation.update(status='PASS',counts={'points':int(v['vertexCount']),'faces':int(v['faceCount']),
          'source_original_ids':v['sourceOriginalFaceIDs'],'source_schema_ids':len(v['sourceSchema']),
          'validated_derived_source_ids':sorted(map(int,v['faceAncestry']))})
    except Exception as exc:
        validation.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
        (outdir/'existing230-validation-error.txt').write_text(str(exc)+'\n')
    input_tuple_equal={}
    for field in TUPLE:
        input_tuple_equal['primary_'+field]=bool(np.array_equal(candidate['face_source_'+field],prepared['face_source_'+field]))
        input_tuple_equal['csr_'+field]=bool(np.array_equal(candidate['quality_star_patch_union_face_source_'+field],prepared['quality_star_patch_union_face_source_'+field]))
    # Only the explicit derived-D authoring-layer projection and 62000204 schema overlay may differ in tuples/schema.
    expected_p=candidate['face_source_layer'].copy();expected_p[layer_report['primary_changed_rows']]=2
    expected_c=candidate['quality_star_patch_union_face_source_layer'].copy();expected_c[layer_report['csr_changed_rows']]=2
    geometry_unchanged=np.array_equal(candidate['points_world_m'].view(np.uint32),prepared['points_world_m'].view(np.uint32)) and np.array_equal(candidate['faces'],prepared['faces'])
    if not np.array_equal(expected_p,prepared['face_source_layer']) or not np.array_equal(expected_c,prepared['quality_star_patch_union_face_source_layer']):
        raise AdapterError('emitted layer arrays contain undeclared tuple changes')
    if not geometry_unchanged: raise AdapterError('authoring compaction changed candidate point/face geometry')
    receipt={'schema':'registered-authoring-view-590-D-593-receipt.v5','status':'PASS' if validation['status']=='PASS' else 'FAIL',
      'scope':'authoring-only source schema and role-layer view','native_steps':0,'physical_admission':False,
      'geometry_admission':False,'physical_candidate_modified':False,
      'inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
      'historical_auxiliary_input':ring1_audit,
      'pinned_commits':{'existing230':'e962da30203adf7a83bf751f5fa169f439006f56'},
      'native_role_source_separation_audit':{'status':'PASS','registered_surface_reader':{'path':PIN_CHILD_READER[0],'sha256':PIN_CHILD_READER[1],'finding':'identity() assigns RegisteredSurface.sourceLayer from the O/I/D reader argument; face source tuples are not used to determine the role namespace.'},'authoring_role_consumer':{'path':PIN_ROLE_CONSUMER[0],'sha256':PIN_ROLE_CONSUMER[1],'finding':'role consumer requires sourceLayer to equal the O/I/D role index (D=2), separate from inherited NPZ source-layer provenance.'}},
      'preserved_590_helper':{'path':str(paths['helper590']),'sha256':sha(paths['helper590'])},
      'candidate_output':{'path':str(candidate_path),'sha256':sha(candidate_path),'counts':counts},
      'filtered_source_output':{'path':str(source_view_path),'sha256':source_view_sha},
      'tuple_provenance':{'path':str(provenance_path),'sha256':sha(provenance_path),
        'all_original_primary_and_CSR_tuple_arrays_retained':True,'face_ancestry_arrays_retained':True,
        'source_layer_arrays_are_original_in_provenance':True,'all_nonlayer_primary_and_CSR_tuple_arrays_unchanged':all(v for k,v in input_tuple_equal.items() if not k.endswith('_layer'))},
      'projection':layer_report,'layer3_parent_provenance_audit':layer3proof,
      'unlayered_derived_provenance_audit':audit_unlayered_derived_provenance(candidate,candidate_schema),
      'canonical14_source_adapter':canonical14proof,'62000204_source_adapter':idproof,
      'geometry_bitwise_unchanged':geometry_unchanged,'primary_csr_tuple_equalities':input_tuple_equal,
      'validation':validation,'source_registry_original_path_sha_retained':True,
      'DERIVED_PATCH_tuples_untouched':True}
    receipt_path=outdir/'actual-existing230-D-validation-receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    # Pin the immutable receipt externally; never put a self-hash field inside it.
    receipt_sha=sha(receipt_path)
    pin={'receipt_path':str(receipt_path),'receipt_sha256':receipt_sha,
      'pin_semantics':'detached SHA-256 of exact receipt bytes; no self-hash field is embedded'}
    (outdir/'actual-existing230-D-validation-receipt.pin.json').write_text(json.dumps(pin,indent=2,sort_keys=True)+'\n')
    if validation['status']!='PASS': raise AdapterError('actual existing230 D validation failed; exact failure retained in output receipt')
    return receipt

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',required=True)
    ap.add_argument('--registry-source',required=True)
    ap.add_argument('--filtered-source',required=True)
    ap.add_argument('--ring1-source',required=False)
    ap.add_argument('--helper590',required=True)
    ap.add_argument('--exporter230',required=True)
    ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    try:
        r=build(args)
        print(json.dumps({'status':r['status'],'receipt_path':str(Path(args.out_dir)/'actual-existing230-D-validation-receipt.json'),
          'receipt_sha256':sha(Path(args.out_dir)/'actual-existing230-D-validation-receipt.json'),
          'validation':r['validation']['status'],'counts':r['candidate_output']['counts'],
          'projection':{'primary':r['projection']['primary_changed_count'],'csr':r['projection']['csr_changed_count']},
          'physical_admission':False,'native_steps':0},sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)},sort_keys=True),file=sys.stderr)
        return 1
if __name__=='__main__': raise SystemExit(main())
