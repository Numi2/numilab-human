"""Narrow reader for the selected v8 current-NHA D311-to-lobe map."""
from __future__ import annotations
import collections
import hashlib
import json
import math
import struct
from pathlib import Path

E=Path('/Users/n/numi-human-resting-evidence-20261005')
V8=E/'native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8'
EXPECTED={
 'composition_report':('composition-report.json','f2fd49d0486e20bdca5ea8638215466f4a59ff59d94f1dffa53c8caa5018b460'),
 'd_map':('final-exact-D311-to-lobes-map.jsonl','726eaf407129577b18eacc7a794a7f8fd20bcb5bfa333820a0ee19a2001747d1'),
 'nha':('final/resting-thorax.nhanatomy','1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc'),
 'receipt':('final/resting-anatomy-receipt.json','118788f2db039f805cff15c0bff7efa8c62ec68595c997564358d14044ee53e8'),
 'manifest':('final/resting-anatomy-manifest.json','0b59c2eb417e339c67ed548b766bbd30b81bec5dedd9de5bea30a7a3c96ddadc'),
 'parent_d_map':('/Users/n/numi-human-resting-evidence-20261005/native-lung-conditioned-final-compose-1078/final-exact-D311-to-lobes-map.jsonl','05ab5b8f6ab02a0f3236495b385bc0002d750be58820117e7f149e64df1e398a'),
 'parent_nha':('/Users/n/numi-human-resting-evidence-20261005/native-lung-conditioned-final-compose-1078/final/resting-thorax.nhanatomy','7f6a8175e3cadb93a6dfc43535467b6896cf1b8bacfd0334df7b5bb03c414e92'),
}
EXPECTED_COUNTS={305:19743,306:21600,307:5514,308:486,309:0}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def _tri_bits(row,face):
 return tuple(tuple(struct.pack('<f',float(c)).hex() for c in row['vertices6'][int(v),:3]) for v in row['faces'][int(face)])

def _key(row):
 return (int(row['lobe_stable_id']),int(row['d_face_row']),int(row['d_source_face_id']),
         int(row['d_run_owner']),int(row['d_run_flag']),int(row['l_face_row']),
         int(row['l_source_face_id']),int(row['l_patch_kind']),str(row['interface_role']))

def _parent_key(row):
 return (int(row['lobe_stable_id']),int(row['d_face_row']),int(row['d_source_face_id']),
         int(row['d_run_owner']),int(row['d_run_flag']),int(row['parent_face_row_1078']),
         int(row['l_source_face_id']),int(row['l_patch_kind']),str(row['interface_role']))

def load_v8_current_dmap(*,base,composition_report_path,final_nha_path,final_nha_sha,final_rows):
 """Verify the single pinned v8 D-map and return the existing 1079 native-map shape."""
 comp_path=Path(composition_report_path).resolve()
 expected_comp=(V8/EXPECTED['composition_report'][0]).resolve()
 if comp_path!=expected_comp or sha(comp_path)!=EXPECTED['composition_report'][1]:
  raise ValueError('v8 composition report path/hash mismatch')
 comp=json.loads(comp_path.read_text())
 if comp.get('schema')!='numi.human.final-lung-selected-composition-dryrun-v1':
  raise ValueError('unsupported v8 composition report')
 if comp.get('status')!='provisional 1105 test case; native and full scans pending':
  raise ValueError('unexpected v8 composition report status')
 outputs=comp.get('outputs',{})
 def check_output(name,key):
  item=outputs.get(name,{})
  path=Path(item.get('path','')).resolve()
  expected_path=(V8/EXPECTED[key][0]).resolve() if not Path(EXPECTED[key][0]).is_absolute() else Path(EXPECTED[key][0]).resolve()
  if path!=expected_path or item.get('sha256')!=EXPECTED[key][1] or not path.is_file() or sha(path)!=EXPECTED[key][1]:
   raise ValueError('v8 output pin mismatch: '+name)
  return path
 dmap_path=check_output('d_map','d_map')
 nha_path=check_output('final_nha','nha')
 receipt_path=check_output('final_receipt','receipt')
 manifest_path=check_output('final_manifest','manifest')
 if nha_path!=Path(final_nha_path).resolve() or EXPECTED['nha'][1]!=final_nha_sha or sha(nha_path)!=final_nha_sha:
  raise ValueError('v8 D map is not bound to caller-selected current NHA')
 if int(comp.get('D_map_count',-1))!=47343:
  raise ValueError('v8 D map count summary mismatch')
 reg=comp.get('D_registration_refresh',{})
 if reg.get('exact_coordinate_and_opposite_winding_checked') is not True or reg.get('row311_face_indices_unchanged') is not True or int(reg.get('registered_reciprocal_face_count',-1))!=47343:
  raise ValueError('v8 D registration summary is incomplete')
 for p,h in comp.get('inputs',{}).items():
  ip=Path(p)
  if not ip.is_file() or sha(ip)!=h: raise ValueError('v8 composition input hash mismatch: '+p)
 parent_map=Path(EXPECTED['parent_d_map'][0]).resolve(); parent_nha=Path(EXPECTED['parent_nha'][0]).resolve()
 if comp['inputs'].get(str(parent_map))!=EXPECTED['parent_d_map'][1] or comp['inputs'].get(str(parent_nha))!=EXPECTED['parent_nha'][1]:
  raise ValueError('v8 report does not bind exact 1078 D-map and NHA parents')
 if sha(parent_map)!=EXPECTED['parent_d_map'][1] or sha(parent_nha)!=EXPECTED['parent_nha'][1]:
  raise ValueError('pinned 1078 source parent changed')
 receipt=json.loads(receipt_path.read_text()); manifest=json.loads(manifest_path.read_text())
 for obj,label in ((receipt,'receipt'),(manifest,'manifest')):
  payload=obj.get('payload',{})
  if payload.get('path')!=str(nha_path) or payload.get('sha256')!=final_nha_sha or int(payload.get('surface_count',-1))!=524:
   raise ValueError('v8 '+label+' does not bind the exact final payload')
 parser=base.load(base.PARSER,'v8_dmap_parent_parser')
 parent_rows=parser.parse_payload(parent_nha)[1]
 if not all(sid in parent_rows and sid in final_rows for sid in (305,306,307,308,309,311)):
  raise ValueError('v8 current/parent NHA lacks a required geometry row')
 parent_map_rows=[json.loads(s) for s in parent_map.read_text().splitlines() if s.strip()]
 parent_lookup=collections.defaultdict(list)
 for r in parent_map_rows:parent_lookup[_key(r)].append(r)
 # Every current full-union row has a unique source-registered 1078 face parent,
 # and both current triangles remain exact Float32 copies of that parent.
 raw=[]; grouped={sid:[] for sid in EXPECTED_COUNTS}; seen_current=set(); seen_d=set()
 counts=collections.Counter(); role_counts=collections.Counter(); area_total=0.0
 with dmap_path.open() as dmap_file:
  for line in dmap_file:
   r=json.loads(line); sid=int(r['lobe_stable_id']);counts[sid]+=1
   if sid not in grouped:raise ValueError('v8 D map contains undeclared lobe')
   if r.get('orientation')!='exact_opposite_winding':raise ValueError('v8 D map lacks exact opposite orientation')
   dfi=int(r['d_face_row']);lfi=int(r['l_face_row']);pfi=int(r.get('parent_face_row_1078',-1))
   if dfi<0 or dfi>=len(final_rows[311]['faces']) or lfi<0 or lfi>=len(final_rows[sid]['faces']) or pfi<0 or pfi>=len(parent_rows[sid]['faces']):
    raise ValueError('v8 D map face row out of bounds')
   k=_key(r)
   if _parent_key(r) not in parent_lookup or len(parent_lookup[_parent_key(r)])!=1:
    raise ValueError('v8 D map current record lacks unique 1078 parent-map identity')
   par=parent_lookup[_parent_key(r)][0]
   if int(par['l_face_row'])!=pfi:raise ValueError('v8 D map parent-row correspondence mismatch')
   if _tri_bits(final_rows[311],dfi)!=_tri_bits(parent_rows[311],int(par['d_face_row'])):
    raise ValueError('v8 D face differs from exact 1078 mapped parent triangle')
   if _tri_bits(final_rows[sid],lfi)!=_tri_bits(parent_rows[sid],pfi):
    raise ValueError('v8 lobe face differs from exact 1078 mapped parent triangle')
   role=str(r['interface_role']); owner=int(r['d_run_owner']);flag=int(r['d_run_flag']);kind=int(r['l_patch_kind'])
   if role=='generated_cut_patch':
    if (owner,flag,kind)!=(sid,1,0):raise ValueError('v8 generated cut patch ownership mismatch')
   elif role=='inherited_opposite_surface':
    if (owner,flag,kind)!=(311,0,1):raise ValueError('v8 inherited patch ownership mismatch')
   else:raise ValueError('unsupported v8 interface role')
   if dfi in seen_d:raise ValueError('v8 D face is mapped more than once')
   if (sid,lfi) in seen_current:raise ValueError('v8 lobe face is mapped more than once')
   seen_d.add(dfi);seen_current.add((sid,lfi))
   out={'lobe':sid,'dface':dfi,'dsrc':int(r['d_source_face_id']),'owner':owner,'flag':flag,
        'lface':lfi,'lsrc':int(r['l_source_face_id']),'kind':kind,'role':role,
        'orientation':'exact_opposite_winding'}
   grouped[sid].append(out);raw.append(r);role_counts[(sid,role)]+=1
   area_total+=float(r['triangle_area_m2'])
 if sum(counts.values())!=47343 or dict(counts)!={k:v for k,v in EXPECTED_COUNTS.items() if v}:
  raise ValueError('v8 D map per-lobe coverage/count mismatch')
 if counts[309]!=0: raise ValueError('v8 map must explicitly leave row309 with zero D interface faces')
 # The one source JSONL need not include zero-row records, so enforce/report zero 309
 # while validating each of the five runtime map declarations below.
 source_rows={}
 d=final_rows[311]
 ds=base.np.full(len(d['faces']),-1,dtype=base.np.int64); do=ds.copy(); df=ds.copy()
 for r in raw:
  i=int(r['d_face_row']);ds[i]=int(r['d_source_face_id']);do[i]=int(r['d_run_owner']);df[i]=int(r['d_run_flag'])
 source_rows[311]={'v':base.np.asarray(d['vertices6'][:,:3],dtype='<f4').copy(),'f':base.np.asarray(d['faces'],dtype=base.np.int64).copy(),
                   'source_face_ids':ds,'run_owner':do,'run_flags':df}
 pairs={}
 for sid in EXPECTED_COUNTS:
  r=final_rows[sid];src=base.np.full(len(r['faces']),-1,dtype=base.np.int64);kinds=base.np.full(len(r['faces']),-1,dtype=base.np.int64)
  for x in grouped[sid]:src[x['lface']]=x['lsrc'];kinds[x['lface']]=x['kind']
  source_rows[sid]={'v':base.np.asarray(r['vertices6'][:,:3],dtype='<f4').copy(),'f':base.np.asarray(r['faces'],dtype=base.np.int64).copy(),
                    'source_face_ids':src,'patch_kinds':kinds}
  pairs[sid]=base.validate_pair_sources(sid,grouped[sid],source_rows,int(EXPECTED_COUNTS[sid]))
 inputs={str(p):sha(p) for p in (comp_path,dmap_path,nha_path,receipt_path,manifest_path,parent_map,parent_nha)}
 inputs.update({str(Path(p)):h for p,h in comp.get('inputs',{}).items()})
 return {'path':str(comp_path),'sha256':sha(comp_path),'map_path':str(dmap_path),'map_sha256':sha(dmap_path),
         'inputs':inputs,'source_rows':source_rows,'pairs':pairs,'declared_pairs':list(EXPECTED_COUNTS),
         'map_count':sum(counts.values()),'counts_by_lobe':dict(counts),'zero_lobes':[sid for sid,n in EXPECTED_COUNTS.items() if n==0],
         'role_counts':{f'{sid}:{role}':n for (sid,role),n in role_counts.items()},
         'registered_area_m2':float(reg['registered_reciprocal_area_m2']),'map_area_sum_m2':area_total,
         'parent_binding':'Every current D/lobe face-map entry resolves uniquely to the pinned 1078 map using current D face/source/owner/flag, parent lobe face row, source ID, kind, and role; current D and lobe Float32 triangles are exactly equal to the corresponding 1078 parent triangles.'}
