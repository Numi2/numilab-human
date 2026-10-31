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
    """Return registry-declared derived D records with source code -1 or 3.

    Code 3 remains BONE provenance. The projection changes only the derived
    record's authoring tuple to role-D code 2; it never reinterprets code 3 as D.
    """
    return sorted(int(sid) for sid,info in schema.items()
        if info.get('role')=='D' and bool(info.get('derived_operation')) and info.get('layer') in (-1,3))

def verify_layer3_provenance(candidate:dict[str,np.ndarray],schema:dict[str,Any])->dict[str,Any]:
    """Authenticate that layer-3 values on derived D records inherit BONE parents."""
    ids=np.asarray(candidate['face_source_original_id'],dtype=np.int64)
    layers=np.asarray(candidate['face_source_layer'],dtype=np.int64)
    cids=np.asarray(candidate['quality_star_patch_union_face_source_original_id'],dtype=np.int64)
    clayers=np.asarray(candidate['quality_star_patch_union_face_source_layer'],dtype=np.int64)
    derived3=[];parent_ids=set();primary_rows=0;csr_rows=0
    for sid_s,info in schema.items():
        if info.get('role')!='D' or not info.get('derived_operation') or info.get('layer')!=3:
            continue
        sid=int(sid_s); derived3.append(sid)
        pr=np.flatnonzero(ids==sid); cr=np.flatnonzero(cids==sid)
        if len(pr)==0 and len(cr)==0: raise AdapterError(f'{sid}: layer-3 derived D registry entry has no candidate tuple rows')
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
    return {'derived_D_layer3_ids':sorted(derived3),'derived_D_layer3_id_count':len(derived3),
      'primary_tuple_rows':primary_rows,'csr_tuple_rows':csr_rows,
      'distinct_parent_source_ids':sorted(parent_ids),'distinct_parent_source_id_count':len(parent_ids),
      'all_layer3_parent_source_ids_registered_as_BONE':True,
      'all_parent_candidate_primary_tuples_verified_layer3':True,
      'semantics':'3 remains inherited BONE source provenance; parent registry role is BONE and its current candidate source tuples are layer 3; authoring role D is separately projected to layer code 2.'}

def project_layers(candidate:dict[str,np.ndarray],schema:dict[str,Any])->tuple[dict[str,np.ndarray],dict[str,Any]]:
    out={k:np.asarray(v).copy() for k,v in candidate.items()}
    selected=projection_ids(schema); source_layer_by_id={sid:int(schema[str(sid)]['layer']) for sid in selected}
    if 'face_source_original_id' not in out or 'face_source_layer' not in out:
        raise AdapterError('missing primary source tuple arrays')
    primary_id=out['face_source_original_id']; primary_layer=out['face_source_layer']
    if primary_id.shape!=primary_layer.shape: raise AdapterError('primary source tuple extent mismatch')
    selected_arr=np.asarray(selected,dtype=np.int64)
    primary_mask=np.isin(primary_id,selected_arr)
    primary_rows=np.flatnonzero(primary_mask).astype(np.int64)
    if len(primary_rows):
        expected=np.fromiter((source_layer_by_id[int(sid)] for sid in primary_id[primary_mask]),dtype=np.int64,count=len(primary_rows))
        if not np.array_equal(primary_layer[primary_mask].astype(np.int64),expected):
            raise AdapterError('primary layer values disagree with pinned source-layer codes for selected D-derived IDs')
    primary_from={str(code):int(np.count_nonzero(primary_layer[primary_mask]==code)) for code in (-1,3)}
    projected_primary_ids=sorted(map(int,np.unique(primary_id[primary_mask])))
    primary_layer[primary_mask]=2
    c_id_key='quality_star_patch_union_face_source_original_id'
    c_layer_key='quality_star_patch_union_face_source_layer'
    if c_id_key not in out or c_layer_key not in out: raise AdapterError('missing typed union source tuples')
    csr_id=out[c_id_key]; csr_layer=out[c_layer_key]
    if csr_id.shape!=csr_layer.shape: raise AdapterError('CSR source tuple extent mismatch')
    csr_mask=np.isin(csr_id,selected_arr)
    csr_rows=np.flatnonzero(csr_mask).astype(np.int64)
    if len(csr_rows):
        expected=np.fromiter((source_layer_by_id[int(sid)] for sid in csr_id[csr_mask]),dtype=np.int64,count=len(csr_rows))
        if not np.array_equal(csr_layer[csr_mask].astype(np.int64),expected):
            raise AdapterError('CSR layer values disagree with pinned source-layer codes for selected D-derived IDs')
    csr_from={str(code):int(np.count_nonzero(csr_layer[csr_mask]==code)) for code in (-1,3)}
    projected_csr_ids=sorted(map(int,np.unique(csr_id[csr_mask])))
    csr_layer[csr_mask]=2
    # Keep non-D roles, non-derived rows, and DERIVED_PATCH records unchanged.
    primary_derived_patch_ids=sorted(int(i) for i in np.unique(primary_id[primary_layer==-1])
        if schema.get(str(int(i)),{}).get('role')=='DERIVED_PATCH')
    csr_derived_patch_ids=sorted(int(i) for i in np.unique(csr_id[csr_layer==-1])
        if schema.get(str(int(i)),{}).get('role')=='DERIVED_PATCH')
    per_source=[]
    for sid in selected:
        per_source.append({'id':sid,'role':schema[str(sid)].get('role'),'source_layer_code':source_layer_by_id[sid],
          'authoring_layer_code':2,'primary_rows':int(np.count_nonzero(primary_id==sid)),
          'csr_rows':int(np.count_nonzero(csr_id==sid))})
    report={
      'rule':'Only schema role D plus nonempty derived_operation entries with source layer code -1 or 3 are projected in the authoring view to role-D code 2. Code 3 remains BONE source provenance in the untouched original tuple archive.',
      'source_registry_projection_ids':selected,
      'projection_id_count':len(selected),
      'primary_changed_rows':primary_rows.tolist(),'primary_changed_count':int(len(primary_rows)),
      'primary_changed_source_ids':projected_primary_ids,'primary_rows_by_original_source_layer':primary_from,
      'csr_changed_rows':csr_rows.tolist(),'csr_changed_count':int(len(csr_rows)),
      'csr_changed_source_ids':projected_csr_ids,'csr_rows_by_original_source_layer':csr_from,
      'per_source_projection':per_source,
      'primary_derived_patch_ids_left_unchanged':primary_derived_patch_ids,
      'csr_derived_patch_ids_left_unchanged':csr_derived_patch_ids,
      'BONE_layer3_primary_rows_unchanged':int(np.count_nonzero((primary_layer==3)&np.isin(primary_id,[int(k) for k,v in schema.items() if v.get('role')=='BONE']))),
      'BONE_layer3_CSR_rows_unchanged':int(np.count_nonzero((csr_layer==3)&np.isin(csr_id,[int(k) for k,v in schema.items() if v.get('role')=='BONE']))),
    }
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
    entry=overlaid[str(DERIVED_ID)]
    original_entry=copy.deepcopy(entry)
    if entry.get('identity_keys')!=['topology_patch_output_face_row','unused']:
        raise AdapterError('62000204 original identity key contract changed')
    entry.update(path=str(source_view_path),sha256=source_view_sha,
      identity_face_id_key='face_source_face_id',
      identity_original_id_key='face_source_original_id',identity_original_id=DERIVED_ID,
      source_geometry_scope='exact D128-filtered 15-face source view; parent patch union retained separately')
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
          'source_schema_layer_code':int(candidate_schema[str(sid)]['layer']),
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
    prepared['native_authoring_schema_adapter']=np.asarray(['registered-authoring-view-590-D-593.v2'])
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
    prov['primary_projected_layer_rows']=np.asarray(layer_report['primary_changed_rows'],dtype=np.int64)
    prov['csr_projected_layer_rows']=np.asarray(layer_report['csr_changed_rows'],dtype=np.int64)
    prov['projected_derived_source_ids']=np.asarray(layer_report['source_registry_projection_ids'],dtype=np.int64)
    provenance_path=outdir/'original-primary-csr-tuples-and-patch-ancestry.npz'
    np.savez_compressed(provenance_path,**prov)
    source_registry_path=outdir/'D128-original-source-registry.json'
    source_registry_path.write_text(json.dumps(candidate_schema,indent=2,sort_keys=True)+'\n')
    special_entry=overlaid[str(DERIVED_ID)]
    overlay={'schema':'registered-authoring-D-source-schema-overlay.v2','source_registry_path':str(paths['registry_source']),
      'source_registry_sha256':REGISTRY_SHA256,'original_source_entry':original_entry,
      'authoring_entry':special_entry,'filtered_source_view_path':str(source_view_path),'filtered_source_view_sha256':source_view_sha,
      'source_identity_rule':'62000204 face_source_face_id=-1 is retained and matched against the filtered D128 view; identity0 is exactly 0..14, identity1=-1; no face-ID invention.',
      'derived_layer_projection':layer_report,'layer3_parent_provenance_audit':layer3proof,
      'derived_D_authoring_projection':'layer 3 is preserved as BONE ancestry in original tuples; role-D derived output tuple rows are projected to authoring layer 2 in a separate authoring view.',
      'source_to_authoring_layer_map':[{'id':sid,'role':'D','source_schema_layer_code':int(candidate_schema[str(sid)]['layer']),'authoring_layer_code':2} for sid in layer_report['source_registry_projection_ids']],
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
    receipt={'schema':'registered-authoring-view-590-D-593-receipt.v2','status':'PASS' if validation['status']=='PASS' else 'FAIL',
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
      'projection':layer_report,'layer3_parent_provenance_audit':layer3proof,'62000204_source_adapter':idproof,
      'geometry_bitwise_unchanged':geometry_unchanged,'primary_csr_tuple_equalities':input_tuple_equal,
      'validation':validation,'source_registry_original_path_sha_retained':True,
      'DERIVED_PATCH_tuples_untouched':True}
    receipt_path=outdir/'actual-existing230-D-validation-receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    receipt['receipt_sha256']=sha(receipt_path)
    # A detached pin receipt avoids a self-hash field inside the pinned JSON.
    (outdir/'actual-existing230-D-validation-receipt.pin.json').write_text(json.dumps({'path':str(receipt_path),'sha256':receipt['receipt_sha256']},indent=2,sort_keys=True)+'\n')
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
