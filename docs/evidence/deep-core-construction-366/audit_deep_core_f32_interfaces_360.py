#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
B=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union'
RAW=B/'exact-currentbone-rebase-329/result-002/batch-024/D-prebuffer-exact-bones-through-024-closed-f64.npz'
D=B/'dnew-vs-i296-inventory-001/typed-D-62000100/run-002/candidate-f32-typed.npz'
MAN=B/'exact-set-union-remaining-013/all185-bank-operand-manifest.json'
PILOT=B/'dnew-vs-i296-inventory-001/typed-D-62000100/run-002/bone-audit-185-native-matrix-001/solid-intersection-pilot-002/report.json'
OUT=P/'deep-core-f32-interface-360'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def quant(x):
 a=np.asarray(x);return {str(q):float(np.quantile(a,q)) for q in [0,.5,.95,.99,1]} if a.size else {}
def main():
 t=time.monotonic();OUT.mkdir(exist_ok=False)
 raw=load(RAW);d=load(D);man=json.loads(MAN.read_text());pilot=json.loads(PILOT.read_text())
 X=d['points_world_m'].astype(np.float64);F=d['faces'];maps=d['source_f64_vertex_rows'].astype(np.int64)
 if len(maps)!=len(X) or maps.min()<0 or maps.max()>=len(raw['points_world_m_f64']):raise ValueError('vertex row domain mismatch')
 prior=raw['points_world_m_f64'][maps];disp=X-prior
 if not np.array_equal(X.astype(np.float32),prior.astype(np.float32)):raise ValueError('emitted D points differ from exact rounding of raw D329')
 rows=[];assets={str(p):sha(p) for p in [RAW,D,MAN,PILOT]}
 lookup={(int(r['body']),int(r['bone'])):r for r in man['operands']}
 for pc in pilot['cases']:
  key=(pc['body'],pc['bone']);m=lookup[key];path=Path(m['operand_path']);z=load(path)
  if sha(path)!=m['operand_sha256']:raise ValueError('operand mismatch')
  assets[str(path)]=m['operand_sha256']
  sid=int(m['source_id']);rr=np.flatnonzero(d['face_source_original_id']==sid)
  ids=d['face_source_face_id'][rr]
  fids=z['face_id'].astype(np.int64);order=np.argsort(fids);at=np.searchsorted(fids[order],ids)
  if (at>=len(fids)).any() or not np.array_equal(fids[order[at]],ids):raise ValueError('unresolved source face')
  tri=z['points_world_m_f64'][z['faces_f64'][order[at]]]
  nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);ln=np.linalg.norm(nn,axis=1)
  if (ln<=0).any():raise ValueError('degenerate source face')
  nn/=ln[:,None];oldface=prior[F[rr]];newface=X[F[rr]]
  olddist=np.einsum('fva,fa->fv',oldface-tri[:,None,0,:],nn)
  newdist=np.einsum('fva,fa->fv',newface-tri[:,None,0,:],nn)
  moved=np.einsum('fva,fa->fv',disp[F[rr]],nn)
  rows.append({'body':key[0],'bone':key[1],'source_id':sid,'fragment_faces':len(rr),'source_plane_pre_emit_abs_distance_m':quant(np.abs(olddist)),'source_plane_post_emit_signed_distance_m':quant(newdist),'rounding_normal_displacement_m':quant(moved),'fragment_faces_with_any_inward_rounded_vertex':int((newdist.min(1)<-1e-12).sum()),'fragment_faces_with_any_outward_rounded_vertex':int((newdist.max(1)>1e-12).sum()),'pilot_bone_outside_D_volume_m3':pc['operations']['bone_outside_D']['signed_volume_sum_m3'],'pilot_bone_outside_D_fraction':pc['bone_outside_D_fraction']})
 np.savez_compressed(OUT/'vertex-rounding-displacement.npz',source_f64_vertex_rows=maps,rounding_displacement_m=disp)
 report={'status':'DIAGNOSTIC_ONLY','source_assets_sha256':assets,'all_emitted_vertices_exactly_equal_raw_F64_rounding':True,'maximum_vertex_rounding_displacement_m':float(np.linalg.norm(disp,axis=1).max()),'vertex_rounding_displacement_m':quant(np.linalg.norm(disp,axis=1)),'four_bank_source_face_fragments':rows,'interpretation':'D is an excluded core. Its inherited bone fragments are intended interfaces. These diagnostics isolate source-plane displacement caused by Float32 rounding, not a containment or physical-contact certificate. No geometry changed; no physical steps run.','physical_admission':False,'elapsed_s':time.monotonic()-t}
 (OUT/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
