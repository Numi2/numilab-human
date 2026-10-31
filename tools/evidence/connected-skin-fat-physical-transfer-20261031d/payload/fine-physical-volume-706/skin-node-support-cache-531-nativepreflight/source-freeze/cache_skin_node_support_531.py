#!/usr/bin/env python3
"""Cache reusable source330 parent support for the stable Skin531 point domain.

The output is only a node support cache. Cell P1 integration, current 143
one-hop adjacency, mass mapping, and physical admission must be regenerated or
verified separately against the final paired geometry.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
H=ROOT/'fine-physical-volume-706'
OUT=H/'skin-node-support-cache-531-nativepreflight/run001'
TARGET=H/'skin-canonical-outer-O-531-root/skin-current-O530-I528-canonical-volume-proposal.npz'
TARGET_RECEIPT=TARGET.parent/'receipt.json'
EXPECTED_TARGET='aba0761c3690bd68d964a1aca472ef70385a70dad16eed75da34ccb01ba58ebe'
LEGACY_VOL=H/'skin010-source004-admission-073-retry001'
LEGACY_ATLAS=H/'current-semantic-atlas-088/skin-semantic-atlas-002/output-001/skin-current-semantic-support.npz'
LEGACY_ATLAS_RECEIPT=LEGACY_ATLAS.parent/'receipt.json'
LEGACY_BUILDER=H/'current-semantic-atlas-088/skin-semantic-atlas-002/build_skin_semantic_atlas_002.py'
REG_OWNER=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
SIDECARS=H/'current-semantic-atlas-088/registered-sidecars-skin-001'
SIDECAR_RECEIPT=SIDECARS/'receipt.json'
PARENT_BASE=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
RADIUS=0.02
TIE=1.0e-8

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def bits3(x):
    a=np.asarray(x,dtype='<f4').reshape(3).copy()
    a[a==0]=0.0
    return a.view('<u4').tobytes()

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def main():
    started=time.monotonic()
    if OUT.exists(): raise RuntimeError(f'refusing existing output directory {OUT}')
    OUT.mkdir(parents=True)
    if sha(TARGET)!=EXPECTED_TARGET: raise RuntimeError('Skin531 candidate pin mismatch')
    tr=json.loads(TARGET_RECEIPT.read_text())
    if tr.get('status')!='PASS_CANONICAL_OUTER_O_SKIN_PARENT_UNIONS_AND_TARGET' or tr.get('candidate_sha256')!=EXPECTED_TARGET or tr.get('physical_admission') is not False or tr.get('native_steps')!=0:
        raise RuntimeError('Skin531 receipt/status/admission mismatch')
    lr=json.loads(LEGACY_ATLAS_RECEIPT.read_text())
    if lr.get('status')!='SOURCE_ONLY_PARTIAL_ATLAS_NOT_PHYSICAL_ADMISSION' or lr.get('physical_admission') is not False:
        raise RuntimeError('legacy 088 atlas receipt mismatch')
    sr=json.loads(SIDECAR_RECEIPT.read_text())
    if sr.get('status')!='reference_registration_ready_geometry_not_admitted' or sr.get('physical_admission') is not False or sr.get('selectedRoles')!=['O','I']:
        raise RuntimeError('230 registration sidecar receipt mismatch')
    with np.load(TARGET,allow_pickle=False) as z:
        points=np.asarray(z['points_world_m'],dtype=np.float32).copy()
        points_f32=np.asarray(z['points_world_m_f32'],dtype=np.float32).copy()
        tets=np.asarray(z['tetrahedra'],dtype=np.int64).copy()
        mats=np.asarray(z['material_indices'],dtype=np.int32).copy()
        typed_paths={r:str(z[f'typed_{r}_candidate_path'].item()) for r in ('O','I')}
        typed_hashes={r:str(z[f'typed_{r}_candidate_sha256'].item()) for r in ('O','I')}
    if not np.array_equal(points.view('<u4'),points_f32.view('<u4')): raise RuntimeError('Skin531 coordinate views differ')
    if points.shape!=(210268,3) or tets.shape!=(729848,4) or len(mats)!=len(tets): raise RuntimeError('Skin531 shape mismatch')
    if set(np.unique(mats).tolist())!={15}: raise RuntimeError('unexpected Skin material set')
    if np.any(tets<0) or np.any(tets>=len(points)) or not np.isfinite(points).all(): raise RuntimeError('Skin531 index/nonfinite error')
    target_keys=[bits3(p) for p in points]
    target_map={k:i for i,k in enumerate(target_keys)}
    if len(target_map)!=len(points): raise RuntimeError('duplicate target F32 point coordinates')
    typed={};surface_role=np.zeros(len(points),dtype=np.uint8)
    for role in ('O','I'):
        path=Path(typed_paths[role])
        if not path.is_file() or sha(path)!=typed_hashes[role]: raise RuntimeError(f'{role} typed point pin mismatch')
        with np.load(path,allow_pickle=False) as z: surf=np.asarray(z['points_world_m_f32'],dtype=np.float32).copy()
        rows=np.asarray([target_map.get(bits3(p),-1) for p in surf],dtype=np.int64)
        if np.any(rows<0) or len(np.unique(rows))!=len(rows): raise RuntimeError(f'{role} typed point crosswalk not exact one-to-one')
        if np.any(surface_role[rows]): raise RuntimeError('O/I typed surfaces overlap')
        surface_role[rows]=1 if role=='O' else 2
        typed[role]=(surf,rows)
    # Reuse historical 073 support only when exact F32 point bits match.
    old_builder=load_module(LEGACY_BUILDER,'skin_atlas_088_for_531_cache')
    old_points,_=old_builder.parse_tetgen_nodes(LEGACY_VOL/'skin.node')
    with np.load(LEGACY_ATLAS,allow_pickle=False) as z: old={k:z[k] for k in z.files}
    if len(old['node_group_offsets'])!=len(old_points)+1: raise RuntimeError('legacy atlas node domain mismatch')
    old_map={bits3(p):r for r,p in enumerate(old_points)}
    if len(old_map)!=len(old_points): raise RuntimeError('duplicate legacy point coordinates')
    n=len(points);node_groups=[defaultdict(set) for _ in range(n)]
    old_row_for_target=np.full(n,-1,dtype=np.int64)
    origin=np.zeros(n,dtype=np.uint8) # 1=230 O/I, 2=legacy 073 point, 3=new source330 query
    distance=np.full(n,np.nan,dtype=np.float64)
    old_tuples=np.asarray(old['group_tuples'],dtype=np.int32)
    for nr,key in enumerate(target_keys):
        oldrow=old_map.get(key)
        if oldrow is None: continue
        old_row_for_target[nr]=oldrow
        origin[nr]=1 if surface_role[nr] else (2 if int(old['node_origin_kind'][oldrow]) else 0)
        d=float(old['node_parent_registration_distance_m'][oldrow])
        if np.isfinite(d): distance[nr]=d
        for slot in range(int(old['node_group_offsets'][oldrow]),int(old['node_group_offsets'][oldrow+1])):
            g=tuple(map(int,old_tuples[int(old['node_group_ids'][slot])]))
            a=int(old['node_group_parent_tet_offsets'][slot]);b=int(old['node_group_parent_tet_offsets'][slot+1])
            node_groups[nr][g].update(map(int,old['node_group_parent_tet_rows'][a:b]))
    # Replay every authenticated 230 source registration row against the new
    # typed O530/I528 point arrays, then merge its complete tied-parent support.
    reg=load_module(REG_OWNER,'reg230_owner_for_skin531_cache')
    parent,index,material_index,parentmeta=reg.load_parent_current(PARENT_BASE)
    if len(parent['materials'])!=28632: raise RuntimeError('source330 parent domain mismatch')
    side_counts={};side_max=0.0
    for role in ('O','I'):
        pts,rows=typed[role];path=SIDECARS/f'registered-{role}.csv';count=0
        with path.open('r',encoding='utf-8',newline='') as f:
            rd=csv.DictReader(f)
            need={'vertex','wx','wy','wz','parent_tet_row','parent_registration_distance_m','parent_registration_tie_count','parent_tied_tet_rows_json','parent_material_index','parent_geometric_donor_body','parent_anatomical_region_id'}
            if not need.issubset(set(rd.fieldnames or [])): raise RuntimeError(f'{role} sidecar schema mismatch')
            for row in rd:
                v=int(row['vertex'])
                if v!=count or v>=len(pts): raise RuntimeError(f'{role} sidecar order/range mismatch')
                sidepoint=np.asarray([float(row[k]) for k in ('wx','wy','wz')],dtype=np.float32)
                if bits3(sidepoint)!=bits3(pts[v]): raise RuntimeError(f'{role} sidecar point mismatch at {v}')
                nr=int(rows[v]);ties=sorted(set(map(int,json.loads(row['parent_tied_tet_rows_json']))))
                selected=int(row['parent_tet_row']);d=float(row['parent_registration_distance_m'])
                if not ties or selected not in ties or not np.isfinite(d) or d>RADIUS+1e-12: raise RuntimeError(f'{role} registration witness invalid at {v}')
                if int(row['parent_registration_tie_count'])!=len(ties)-1: raise RuntimeError(f'{role} tie count mismatch at {v}')
                claimed=(int(row['parent_material_index']),int(row['parent_geometric_donor_body']),int(row['parent_anatomical_region_id']))
                if tuple(int(parent[k][selected]) for k in ('materials','donors','regions'))!=claimed: raise RuntimeError(f'{role} selected source tuple mismatch at {v}')
                for pr in ties:
                    if pr<0 or pr>=len(parent['materials']): raise RuntimeError(f'{role} tied source row out of range')
                    g=(int(parent['materials'][pr]),int(parent['donors'][pr]),int(parent['regions'][pr]))
                    node_groups[nr][g].add(pr)
                origin[nr]=1;distance[nr]=max(d,float(distance[nr]) if np.isfinite(distance[nr]) else 0.0)
                side_max=max(side_max,d);count+=1
        if count!=len(pts): raise RuntimeError(f'{role} sidecar incomplete: {count}/{len(pts)}')
        side_counts[role]=count
    # Query only unsupported target points. Exact legacy and registered O/I
    # parent supports are reused, so topology-only changes cost no point search.
    query_rows=np.asarray([r for r,g in enumerate(node_groups) if not g],dtype=np.int64)
    query_fail=Counter();query_witness=[];query_dist=[]
    for qi,nr in enumerate(query_rows):
        try: q=reg.closest_same_material(parent,index,material_index,15,points[int(nr)].astype(np.float64),RADIUS,TIE)
        except Exception as exc:
            key=type(exc).__name__+':'+str(exc).split(' m ')[0];query_fail[key]+=1
            if len(query_witness)<32: query_witness.append({'target_node_row':int(nr),'xyz_f32_m':points[int(nr)].astype(float).tolist(),'error':str(exc)})
            continue
        ties=sorted(set(map(int,q['ties'])));d=float(q['distance'])
        if not ties or not np.isfinite(d) or d>RADIUS+1e-12: raise RuntimeError(f'new-node registration bound failure at row {nr}')
        for pr in ties:
            g=(int(parent['materials'][pr]),int(parent['donors'][pr]),int(parent['regions'][pr]))
            if g[0]!=15: raise RuntimeError(f'non-Skin group from same-material query at row {nr}')
            node_groups[int(nr)][g].add(pr)
        origin[int(nr)]=3;distance[int(nr)]=d;query_dist.append(d)
        if (qi+1)%1000==0: print(json.dumps({'stage':'new_node_queries','done':qi+1,'total':len(query_rows),'failures':sum(query_fail.values())}),flush=True)
    # Deterministic CSR encoding with full source-parent and stable-face unions.
    groups=sorted({g for ng in node_groups for g in ng});gid={g:i for i,g in enumerate(groups)}
    ng_off=[0];ng_ids=[];parent_off=[0];parent_rows=[];face_off=[0];face_rows=[]
    counts=Counter();unsupported=0;multi=0
    for nr,ng in enumerate(node_groups):
        if not ng: unsupported+=1
        if len(ng)>1: multi+=1
        for g in sorted(ng):
            prs=sorted(set(map(int,ng[g])))
            if not prs: raise RuntimeError(f'empty parent support at node {nr}')
            if any((int(parent['materials'][x]),int(parent['donors'][x]),int(parent['regions'][x]))!=g for x in prs): raise RuntimeError(f'group-parent mismatch at node {nr}')
            frs=sorted(set(int(parent['faces'][x]) for x in prs))
            if not frs: raise RuntimeError(f'missing stable-face union at node {nr}')
            ng_ids.append(gid[g]);parent_rows.extend(prs);parent_off.append(len(parent_rows));face_rows.extend(frs);face_off.append(len(face_rows))
        ng_off.append(len(ng_ids));counts[len(ng)]+=1
    arrays={
      'points_world_m_f32':points,'target_node_row_to_legacy_073_node_row':old_row_for_target,
      'node_origin_kind':origin,'node_registered_surface_role_mask':surface_role,
      'node_parent_registration_distance_m':distance,'group_tuples':np.asarray(groups,dtype=np.int32).reshape((-1,3)),
      'node_group_offsets':np.asarray(ng_off,dtype=np.int64),'node_group_ids':np.asarray(ng_ids,dtype=np.int32),
      'node_group_parent_tet_offsets':np.asarray(parent_off,dtype=np.int64),'node_group_parent_tet_rows':np.asarray(parent_rows,dtype=np.int32),
      'node_group_stable_face_offsets':np.asarray(face_off,dtype=np.int64),'node_group_stable_face_rows':np.asarray(face_rows,dtype=np.uint32)}
    node_path=OUT/'skin-node-source330-support.npz';np.savez_compressed(node_path,**arrays)
    with np.load(node_path,allow_pickle=False) as z:
        for k,v in arrays.items():
            if not np.array_equal(z[k],v,equal_nan=True): raise RuntimeError(f'serialized array mismatch {k}')
    inputs=[Path(__file__),TARGET,TARGET_RECEIPT,LEGACY_VOL/'skin.node',LEGACY_VOL/'receipt.json',LEGACY_ATLAS,LEGACY_ATLAS_RECEIPT,LEGACY_BUILDER,REG_OWNER,SIDECAR_RECEIPT,SIDECARS/'registered-O.csv',SIDECARS/'registered-I.csv',Path(typed_paths['O']),Path(typed_paths['I']),PARENT_BASE,Path(str(PARENT_BASE)+'.tets.csv'),Path(str(PARENT_BASE)+'.labels.csv'),Path(str(PARENT_BASE)+'.nodes.csv'),Path(str(PARENT_BASE)+'.weights.csv')]
    status='PASS_SOURCE_ONLY_REUSABLE_POINT_SUPPORT_CACHE' if unsupported==0 and not query_fail else 'PARTIAL_SOURCE_ONLY_POINT_SUPPORT_CACHE_WITH_UNRESOLVED_NODES'
    out={
      'schema':'numi.human.skin-current-source330-node-support-cache.v1','status':status,'physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN',
      'meaning':'Reusable exact-F32 current Skin point support only. Current-cell P1 integration and 143 one-hop adjacency must be regenerated against the final paired geometry. Not a complete cell semantic support, mass allocation, or physical admission.',
      'geometry_target':{'path':str(TARGET),'sha256':sha(TARGET),'receipt_path':str(TARGET_RECEIPT),'receipt_sha256':sha(TARGET_RECEIPT),'nodes':len(points),'cells':len(tets),'material_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(mats,return_counts=True))}},
      'point_reuse':{'legacy_073_point_count':len(old_points),'legacy_exact_matches':int(np.count_nonzero(old_row_for_target>=0)),'legacy_points_retired':int(len(old_points)-np.count_nonzero(old_row_for_target>=0)),'target_points_with_legacy_support':int(np.count_nonzero(old_row_for_target>=0)),'new_point_queries':len(query_rows),'typed_O_points':side_counts['O'],'typed_I_points':side_counts['I'],'typed_OI_exact_misses':0,'typed_OI_overlap':0,'surface_max_registration_distance_m':side_max},
      'support':{'source330_parent_rows':len(parent['materials']),'group_count':len(groups),'node_group_slots':len(ng_ids),'group_slots_per_node':{str(k):int(v) for k,v in sorted(counts.items())},'multi_group_nodes':multi,'unsupported_nodes':unsupported,'query_failures':sum(query_fail.values()),'query_failure_reasons':dict(query_fail),'query_failure_witnesses':query_witness,'query_distance_max_m':float(max(query_dist,default=0.0)),'query_distance_p99_m':float(np.quantile(query_dist,.99)) if query_dist else 0.0,'node_group_parent_rows':len(parent_rows),'node_group_stable_face_rows':len(face_rows)},
      'source330_parent_identity':parentmeta,'input_sha256':{str(p):sha(p) for p in inputs},
      'outputs':{'node_support_npz':str(node_path),'node_support_sha256':sha(node_path),'node_support_size_bytes':node_path.stat().st_size},'elapsed_s':time.monotonic()-started}
    (OUT/'receipt.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    (OUT/'query-failures.json').write_text(json.dumps({'failures':sum(query_fail.values()),'reasons':dict(query_fail),'witnesses':query_witness,'physical_admission':False},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':status,'geometry_target':out['geometry_target'],'point_reuse':out['point_reuse'],'support':out['support'],'outputs':out['outputs'],'elapsed_s':out['elapsed_s']},sort_keys=True),flush=True)
    if status!='PASS_SOURCE_ONLY_REUSABLE_POINT_SUPPORT_CACHE': raise RuntimeError(status)
if __name__=='__main__': main()

