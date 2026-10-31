#!/usr/bin/env python3
"""Versioned authoring-only D adapter layered on the immutable 590 helper.

This changes only authoring tuple/schema copies. It never edits physical D,
volume, O/I, registration, or published files. Derived role-D layer sentinels
are projected only under the pinned source-registry rule recorded in the output.
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
    """Only registry-declared same-role D derived records can map -1 to 2."""
    return sorted(int(sid) for sid,info in schema.items()
        if info.get('role')=='D' and info.get('layer')==-1 and bool(info.get('derived_operation')))

def project_layers(candidate:dict[str,np.ndarray],schema:dict[str,Any])->tuple[dict[str,np.ndarray],dict[str,Any]]:
    out={k:np.asarray(v).copy() for k,v in candidate.items()}
    selected=set(projection_ids(schema))
    if not selected: raise AdapterError('pinned registry contains no D-derived layer sentinel rule')
    if 'face_source_original_id' not in out or 'face_source_layer' not in out:
        raise AdapterError('missing primary source tuple arrays')
    primary_id=out['face_source_original_id']; primary_layer=out['face_source_layer']
    if primary_id.shape!=primary_layer.shape: raise AdapterError('primary source tuple extent mismatch')
    primary_mask=np.isin(primary_id,list(selected)) & (primary_layer==-1)
    primary_rows=np.flatnonzero(primary_mask).astype(np.int64)
    projected_primary_ids=sorted(map(int,np.unique(primary_id[primary_mask])))
    if np.any(np.isin(primary_id[primary_mask],list(selected))==False):
        raise AdapterError('primary selection escaped pinned D-derived registry IDs')
    primary_layer[primary_mask]=2
    c_id_key='quality_star_patch_union_face_source_original_id'
    c_layer_key='quality_star_patch_union_face_source_layer'
    if c_id_key not in out or c_layer_key not in out: raise AdapterError('missing typed union source tuples')
    csr_id=out[c_id_key]; csr_layer=out[c_layer_key]
    if csr_id.shape!=csr_layer.shape: raise AdapterError('CSR source tuple extent mismatch')
    csr_mask=np.isin(csr_id,list(selected)) & (csr_layer==-1)
    csr_rows=np.flatnonzero(csr_mask).astype(np.int64)
    projected_csr_ids=sorted(map(int,np.unique(csr_id[csr_mask])))
    csr_layer[csr_mask]=2
    # Do not touch DERIVED_PATCH or any unregistered/other-role sentinel.
    primary_derived_patch_ids=sorted(int(i) for i in np.unique(primary_id[primary_layer==-1])
        if schema.get(str(int(i)),{}).get('role')=='DERIVED_PATCH')
    csr_derived_patch_ids=sorted(int(i) for i in np.unique(csr_id[csr_layer==-1])
        if schema.get(str(int(i)),{}).get('role')=='DERIVED_PATCH')
    report={
      'rule':'schema role D AND schema layer -1 AND nonempty derived_operation; project matching primary/CSR layer tuple values -1 to authoring role layer 2',
      'source_registry_projection_ids':sorted(selected),
      'primary_changed_rows':primary_rows.tolist(),
      'primary_changed_count':int(len(primary_rows)),
      'primary_changed_source_ids':projected_primary_ids,
      'csr_changed_rows':csr_rows.tolist(),
      'csr_changed_count':int(len(csr_rows)),
      'csr_changed_source_ids':projected_csr_ids,
      'primary_derived_patch_ids_left_unchanged':primary_derived_patch_ids,
      'csr_derived_patch_ids_left_unchanged':csr_derived_patch_ids,
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
    paths={k:Path(getattr(args,k)) for k in ('candidate','registry_source','filtered_source','ring1_source','helper590','exporter230')}
    expected={'candidate':INPUT_SHA256,'registry_source':REGISTRY_SHA256,'filtered_source':FILTERED_SOURCE_SHA256,
      'ring1_source':RING1_SHA256,'helper590':PIN_590_HELPER_SHA256,'exporter230':PIN_230_EXPORTER_SHA256}
    for name,path in paths.items():
        if not path.is_file(): raise AdapterError(f'{name} file unavailable: {path}')
        actual=sha(path)
        if actual!=expected[name]: raise AdapterError(f'{name} pinned SHA mismatch: {actual} != {expected[name]}')
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
    prepared['native_authoring_schema_adapter']=np.asarray(['registered-authoring-view-590-D-593.v1'])
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
    overlay={'schema':'registered-authoring-D-source-schema-overlay.v1','source_registry_path':str(paths['registry_source']),
      'source_registry_sha256':REGISTRY_SHA256,'original_source_entry':original_entry,
      'authoring_entry':entry,'filtered_source_view_path':str(source_view_path),'filtered_source_view_sha256':source_view_sha,
      'source_identity_rule':'62000204 face_source_face_id=-1 is retained and matched against the filtered D128 view; identity0 is exactly 0..14, identity1=-1; no face-ID invention.',
      'derived_layer_projection':layer_report,'DERIVED_PATCH_layer_tuples_modified':False,
      'raw_geometry_sha256':entry['raw_geometry_sha256'],'physical_candidate_modified':False}
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
    # Only the explicit layer projection and 62000204 schema overlay may differ in tuples/schema.
    expected_p=candidate['face_source_layer'].copy();expected_p[layer_report['primary_changed_rows']]=2
    expected_c=candidate['quality_star_patch_union_face_source_layer'].copy();expected_c[layer_report['csr_changed_rows']]=2
    geometry_unchanged=np.array_equal(candidate['points_world_m'].view(np.uint32),prepared['points_world_m'].view(np.uint32)) and np.array_equal(candidate['faces'],prepared['faces'])
    if not np.array_equal(expected_p,prepared['face_source_layer']) or not np.array_equal(expected_c,prepared['quality_star_patch_union_face_source_layer']):
        raise AdapterError('emitted layer arrays contain undeclared tuple changes')
    if not geometry_unchanged: raise AdapterError('authoring compaction changed candidate point/face geometry')
    receipt={'schema':'registered-authoring-view-590-D-593-receipt.v1','status':'PASS' if validation['status']=='PASS' else 'FAIL',
      'scope':'authoring-only source schema and role-layer view','native_steps':0,'physical_admission':False,
      'geometry_admission':False,'physical_candidate_modified':False,
      'inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
      'pinned_commits':{'existing230':'e962da30203adf7a83bf751f5fa169f439006f56'},
      'preserved_590_helper':{'path':str(paths['helper590']),'sha256':sha(paths['helper590'])},
      'candidate_output':{'path':str(candidate_path),'sha256':sha(candidate_path),'counts':counts},
      'filtered_source_output':{'path':str(source_view_path),'sha256':source_view_sha},
      'tuple_provenance':{'path':str(provenance_path),'sha256':sha(provenance_path),
        'all_original_primary_and_CSR_tuple_arrays_retained':True,'face_ancestry_arrays_retained':True,
        'source_layer_arrays_are_original_in_provenance':True,'all_nonlayer_primary_and_CSR_tuple_arrays_unchanged':all(v for k,v in input_tuple_equal.items() if not k.endswith('_layer'))},
      'projection':layer_report,'62000204_source_adapter':idproof,
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
    ap.add_argument('--ring1-source',required=True)
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
