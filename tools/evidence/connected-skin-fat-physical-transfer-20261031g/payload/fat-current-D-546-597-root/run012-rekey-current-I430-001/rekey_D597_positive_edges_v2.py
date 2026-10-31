#!/usr/bin/env python3
"""Exact-F32 rekey of prior measured positive D edges to current D/I430 Fat."""
from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
R=H/'fat-current-D-546-597-root'
OUT=R/'run012-rekey-current-I430-001/result-001'
SRC=R/'run003-measured-positive-subset-001/allowed-D-positive-gain-edge-map.npz'
SEED=R/'run002-preflight-001/identity-seed-D.npz'
D=R/'run009-typed-bind-D597-current-D558-v1/candidate-D-typed-current.npz'
FAT=H/'paired-I-current620-root/canonical621/fat-current-I430-canonical-volume-proposal.npz'
PINS={'source_map':'89dc816c1fad4c5650a5b19ec0731e26999dd7f9287d4ee9b6ab4176a4b1782a',
'seed':'42d893cd14eb317c0ef87a1acc4b052efcc034216df0776468d097ec4e9060ad',
'D':'97eb4720ebb09d62659aa03f6c4f86482f9cf0e949101f636f6d1fff0e1daa89',
'Fat':'8b90195f20b12f35aa9f44b2054abb9585e5033e26e3bb8b7b5242d31279c39d'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def edge_faces(faces):
 out={}
 for i,(a,b,c) in enumerate(np.asarray(faces,np.int64)):
  for u,v in ((a,b),(b,c),(c,a)):
   out.setdefault(tuple(sorted((int(u),int(v)))),[]).append(i)
 return out
def main():
 start=time.monotonic()
 if OUT.exists():raise FileExistsError(f'refusing existing {OUT}')
 OUT.mkdir(parents=True)
 paths={'source_map':SRC,'seed':SEED,'D':D,'Fat':FAT}
 hashes={k:sha(p) for k,p in paths.items()}
 for k,want in PINS.items():
  if hashes[k]!=want:raise ValueError(f'{k} SHA mismatch {hashes[k]} != {want}')
 old,cur,vol,mp=[load(paths[k]) for k in ('seed','D','Fat','source_map')]
 p0=np.asarray(old['points_world_m_f32'],np.float32);p=np.asarray(cur['points_world_m_f32'],np.float32)
 if p0.shape!=p.shape or not np.array_equal(p0,p):raise ValueError('source/current D F32 point arrays differ')
 if not np.array_equal(np.asarray(old['points_world_m'],np.float64),np.asarray(cur['points_world_m'],np.float64)):raise ValueError('source/current D F64 point arrays differ')
 f=np.asarray(cur['faces'],np.int64);vp=np.asarray(vol['points_world_m_f32'],np.float32)
 d2v=np.asarray(vol['fat_current_target_rebind_D_point_to_volume_rows'],np.int64)
 if len(d2v)!=len(p) or np.any(d2v<0) or np.any(d2v>=len(vp)) or not np.array_equal(p,vp[d2v]):raise ValueError('current D-to-I430 Fat F32 point crosswalk fails')
 inc=edge_faces(f);source_edges=np.asarray(mp['D_edge_rows'],np.int64)
 oe=np.asarray(mp['D_allowed_orientation_edge_rows'],np.int64);drops=np.asarray(mp['D_allowed_collapsed_point_rows'],np.int64);keeps=np.asarray(mp['D_allowed_retained_point_rows'],np.int64)
 gain=np.asarray(mp['D_orientation_native_bad_cell_gain'],np.int64);before=np.asarray(mp['D_orientation_native_bad_before'],np.int64);after=np.asarray(mp['D_orientation_native_bad_after'],np.int64)
 delta=np.asarray(mp['D_orientation_exact_det6_delta_integer']).astype(str)
 if len(source_edges)!=278 or len(oe)!=len(drops) or len(drops)!=len(keeps) or len(drops)!=len(gain):raise ValueError('source map dimensions fail pinned contract')
 op={}
 for i,(e,d,k) in enumerate(zip(oe,drops,keeps)):
  key=tuple(sorted(map(int,e)))
  if key in op or key!=tuple(sorted((int(d),int(k)))):raise ValueError('duplicate or inconsistent source orientation')
  op[key]=(int(d),int(k),int(gain[i]),int(before[i]),int(after[i]),delta[i])
 active=[];seed=[];drop=[];keep=[];gain2=[];before2=[];after2=[];delta2=[];length=[];coverage=[];absent=[]
 for e0 in source_edges:
  e=tuple(sorted(map(int,e0)));d,k,g,b,a,dt=op[e]
  bits=np.asarray(p[np.asarray(e,np.int64)],np.float32).view(np.uint32).tolist()
  common={'edge_D_rows':list(e),'sorted_endpoint_f32_bits_u32':bits,'historical_orientation':[d,k],'historical_gain_not_current':g}
  if e not in inc:
   absent.append(e);coverage.append({**common,'status':'already_absent_from_current_D_edge_set'});continue
  fs=inc[e]
  if len(fs)!=2:raise ValueError(f'current edge {e} has {len(fs)} incident faces')
  l=float(np.linalg.norm(p[e[0]].astype(np.float64)-p[e[1]].astype(np.float64)))
  if not 0<l<=0.002:raise ValueError(f'current edge outside map cap: {e} length={l}')
  active.append(e);seed.append(min(fs));drop.append(d);keep.append(k);gain2.append(g);before2.append(b);after2.append(a);delta2.append(dt);length.append(l)
  coverage.append({**common,'status':'active_after_exact_F32_rekey','current_seed_face_row':min(fs),'current_incident_faces':fs,'current_length_m':l})
 if len(active)!=126 or len(absent)!=152 or len(coverage)!=278:raise ValueError(f'expected 126 active/152 absent/278 total, got {len(active)}/{len(absent)}/{len(coverage)}')
 outmap=OUT/'allowed-D-remaining-positive-edge-proposals.npz'
 np.savez_compressed(outmap,D_edge_rows=np.asarray(active,np.int64).reshape(-1,2),D_seed_face_rows=np.asarray(seed,np.int64),
 D_allowed_orientation_edge_rows=np.asarray(active,np.int64).reshape(-1,2),D_allowed_collapsed_point_rows=np.asarray(drop,np.int64),D_allowed_retained_point_rows=np.asarray(keep,np.int64),
 D_source_current_edge_census_counts=np.full(len(active),2,np.int32),D_edge_length_m=np.asarray(length,np.float64),D_seed_face_qface=np.zeros(len(active),np.float64),
 D_orientation_native_bad_cell_gain=np.asarray(gain2,np.int32),D_orientation_native_bad_before=np.asarray(before2,np.int32),D_orientation_native_bad_after=np.asarray(after2,np.int32),
 D_orientation_exact_det6_delta_integer=np.asarray(delta2,dtype='U256'),D_orientation_affected_cell_offsets=np.zeros(len(active)+1,np.int64),D_orientation_affected_cell_rows=np.empty(0,np.int64),
 source_map_sha256=np.asarray([hashes['source_map']],dtype='U64'),source_D_seed_sha256=np.asarray([hashes['seed']],dtype='U64'),
 current_D_sha256=np.asarray([hashes['D']],dtype='U64'),current_Fat_sha256=np.asarray([hashes['Fat']],dtype='U64'),schema=np.asarray(['h706.current-D597.positive-map-exact-F32-rekey.v1'],dtype='U64'))
 ledger=OUT/'edge-coverage-frontier.jsonl'
 with ledger.open('w') as o:
  for row in coverage:o.write(json.dumps(row,sort_keys=True)+'\n')
 receipt={'schema':'h706.current-D-positive-proposal-rekey-receipt.v1','status':'PASS_EXACT_F32_REKEYED_POSITIVE_PROPOSALS_SOURCE_ONLY',
 'inputs':{k:{'path':str(v),'sha256':hashes[k]} for k,v in paths.items()},
 'crosswalks':{'D seed/current point arrays bit_identical':True,'current D to current I430 Fat F32 point rows bit_identical':True,'D_point_count':len(p)},
 'coverage':{'historical_positive_map_edges':278,'active_current_D_edges':len(active),'already_absent_current_D_edges':len(absent),'edge_ledger_rows':len(coverage),'ledger_path':str(ledger),'ledger_sha256':sha(ledger),'edge_key':'exact sorted Float32 endpoint bit tuples plus current D row pair','historical_100k_preflight_incomplete':True,'no_resume_cursor_claimed':True},
 'proposal_map':{'path':str(outmap),'sha256':sha(outmap),'active_edges':len(active),'current_seed_faces_rebuilt':True},
 'historical_native_bad_gains_are_not_current_recheck_results':True,'full_current_Fat_stars_rechecked':False,'volume_transfer':'NOT_RUN','mass_mapping':'NOT_RUN','physical_admission':False,'native_steps':0,'elapsed_seconds':time.monotonic()-start}
 rp=OUT/'receipt.json';rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':receipt['status'],'active_edges':len(active),'absent_edges':len(absent),'map_sha256':sha(outmap),'ledger_sha256':sha(ledger),'receipt_sha256':sha(rp),'elapsed_s':receipt['elapsed_seconds']},indent=2))
if __name__=='__main__':main()

