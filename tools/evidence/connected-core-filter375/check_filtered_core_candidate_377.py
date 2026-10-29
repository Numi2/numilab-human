#!/usr/bin/env python3
"""Check signed core-domain change before admitting a connected repair."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');OUT=P/'deep-core-change-coverage-377'
OLD=P/'deep-core-void-coalescence-366/D-closed-csg-only.npz';NEW=P/'deep-core-connected-patch-filter-375/D-closed-csg-only.npz';CAND=P/'deep-core-connected-patch-filter-375/candidate-f32-typed.npz';REC=P/'deep-core-connected-patch-filter-375/receipt.json';MAN=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def main():
 t=time.monotonic();OUT.mkdir(exist_ok=False);sys.path.insert(0,str(P.parent/'authoring-deps'));import manifold3d as mf
 def solid(path):
  with np.load(path,allow_pickle=False) as z:p=z['points_world_m'].astype(np.float64);f=z['faces'].astype(np.uint64)
  m=mf.Manifold(mf.Mesh64(vert_properties=p,tri_verts=f))
  if m.status()!=mf.Error.NoError:raise RuntimeError(str(m.status()))
  return m
 old=solid(OLD);new=solid(NEW);lost=old-new;added=new-old
 if lost.status()!=mf.Error.NoError or added.status()!=mf.Error.NoError:raise RuntimeError('core difference Boolean failed')
 rec=json.loads(REC.read_text());manifest=json.loads(MAN.read_text());banks={(int(r['body']),int(r['bone'])):r for r in manifest['operands']}
 with np.load(CAND,allow_pickle=False) as z:
  rows=z['face_source_original_id']==62000200;off=z['face_ancestry_offsets'];sel=np.concatenate([np.arange(off[r],off[r+1]) for r in np.flatnonzero(rows)]);orig=z['face_ancestry_original_ids'][sel];ib=z['face_ancestry_identity0'][sel];ik=z['face_ancestry_identity1'][sel]
  changed_source_ids=set(map(int,orig))
  schema=json.loads(str(z['face_source_identity_schema_by_original_id_json'][0]))
  selected=[r for r in manifest['operands'] if int(r['source_id']) in changed_source_ids]
  for r in selected:
   ss=schema[str(int(r['source_id']))]
   if (int(ss['body']),int(ss['bone']))!=(int(r['body']),int(r['bone'])):raise ValueError('typed source-id to bank metadata mismatch')
  changedbanks=sorted(banks) # Whole source-bank manifest: do not assume ancestry exhausts nearby occupants.
 with np.load(P/'deep-core-connected-patch-filter-375/geometry-checkpoint.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
 patch_boxes=[]
 with np.load(P/'deep-core-void-coalescence-366/candidate-f32-typed.npz',allow_pickle=False) as z:px=z['points_world_m'];pf=z['faces']
 for q in rec['accepted_components']:
  tri=px[pf[np.asarray(q['removed_rows'],np.int64)]].reshape(-1,3).astype(np.float64)
  patch_boxes.append((int(q['component']),tri.min(0),tri.max(0)))
 cases=[];unsafe=set();witnesses=[]
 for key in changedbanks:
  r=banks[key];path=Path(r['operand_path'])
  if sha(path)!=r['operand_sha256']:raise ValueError('bank hash mismatch')
  with np.load(path,allow_pickle=False) as z:bp=z['points_world_m_f64'];bf=z['faces_f64'].astype(np.uint64)
  b=mf.Manifold(mf.Mesh64(vert_properties=np.array(bp,np.float64,order='C'),tri_verts=np.array(bf,np.uint64,order='C')))
  removedbone=mf.Manifold.batch_boolean([b,lost],mf.OpType.Intersect)
  if b.status()!=mf.Error.NoError or removedbone.status()!=mf.Error.NoError:raise ValueError('changed-bank Boolean failed')
  volume=float(removedbone.volume());matches=set()
  for i,part in enumerate(removedbone.decompose()):
   if part.volume()<=0:continue
   mesh=part.to_mesh64();pp=np.asarray(mesh.vert_properties)[:,:3];lo=pp.min(0);hi=pp.max(0)
   hit=[ci for ci,blo,bhi in patch_boxes if np.all(bhi>=lo-1e-10) and np.all(hi+1e-10>=blo)]
   matches.update(hit);unsafe.update(hit)
   witnesses.append({'body':key[0],'bone':key[1],'component':i,'volume_m3':float(part.volume()),'bbox_min_m':lo.tolist(),'bbox_max_m':hi.tolist(),'overlapping_patch_components':hit})
  cases.append({'body':key[0],'bone':key[1],'lost_core_intersect_bone_volume_m3':volume,'overlapping_patch_components':sorted(matches)})
 r={'status':'BOUNDED_CHANGED_DOMAIN_SOLID_DIAGNOSTIC','inputs':{str(p):sha(p) for p in (OLD,NEW,CAND,REC,MAN)},'old_core_volume_m3':float(old.volume()),'new_core_volume_m3':float(new.volume()),'lost_excluded_core_volume_m3':float(lost.volume()),'added_excluded_core_volume_m3':float(added.volume()),'source_ancestry_selected_changed_banks':changedbanks,'bank_cases':cases,'positive_lost_bone_witnesses':witnesses,'unsafe_patch_components':sorted(unsafe),'manifest_banks_checked':len(changedbanks),'sum_lost_core_intersect_changed_bones_m3':sum(c['lost_core_intersect_bone_volume_m3'] for c in cases),'maximum_sampled_patch_deviation_m':rec['maximum_sampled_patch_deviation_m'],'interpretation':'F64 solid diagnostic of construction changes, not exact all185 containment or physical admission. Positive lost-core/bone intersection is a newly exposed bone volume in the fat domain and requires review.','physical_admission':False,'elapsed_s':time.monotonic()-t}
 # Recheck new faces against the final filtered topology; removed operations restored old faces.
 import importlib.util
 fp=P/'geometry-worker/D-connectivity-218/limiting-O-edge-audit-001/quality-failure-inventory-001/contraction-batch-001/shape-repair-batch-005/contraction-001/authoring-source-chain/local_float32_edge_flip.py'
 spec=importlib.util.spec_from_file_location('exact_flip_owner376',fp);fm=importlib.util.module_from_spec(spec);spec.loader.exec_module(fm)
 with np.load(CAND,allow_pickle=False) as z:cx=z['points_world_m'];cf=z['faces'];newrows=np.flatnonzero(z['face_source_original_id']==62000200)
 owner=fm.Flips(cx,cf);pairs=owner.pairs(set(map(int,newrows)))
 r['new_face_exact_pair_count_against_final_filtered_surface']=None if pairs is None else int(pairs)
 r['new_face_rows_checked']=len(newrows);r['exact_owner_sha256']=sha(fp)
 r['global_self_certificate']='Parent366 retained face subset has zero pairs; all newly introduced filtered faces tested against final current surface.'
 r['elapsed_s']=time.monotonic()-t
 (OUT/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
