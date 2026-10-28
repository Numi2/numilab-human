#!/usr/bin/env python3
"""Offline connected skin-envelope topology reconstruction, with explicit derivation."""
from pathlib import Path
from emit_f32 import emit
import argparse, hashlib, importlib.util, json, sys,time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'authoring-deps'));import manifold3d as mf

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m

def close_for_boolean(X,F):
 # F is outward oriented here. Cap only as a temporary boolean construction.
 E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]);U,inv,c=np.unique(np.sort(E,axis=1),axis=0,return_inverse=True,return_counts=True)
 assert np.all(c<=2)
 boundary=E[c[inv]==1];nextv={int(a):int(b) for a,b in boundary};assert len(nextv)==len(boundary)
 loops=[];todo=set(nextv)
 while todo:
  start=min(todo);loop=[start];v=nextv[start]
  while v!=start:
   assert v not in loop;loop.append(v);v=nextv[v]
  todo.difference_update(loop);loops.append(loop)
 V=list(np.asarray(X,dtype=float));caps=[]
 for loop in loops:
  center=len(V);V.append(np.mean(X[loop].astype(float),axis=0))
  for a,b in zip(loop,loop[1:]+loop[:1]):caps.append([b,a,center])
 return np.array(V),np.vstack([F,np.array(caps)]),boundary,loops

def from_mesh(V,F,face_ids=None):
 kw={'vert_properties':np.array(V,dtype=np.float64,order="C",copy=True),'tri_verts':np.array(F,dtype=np.uint64,order="C",copy=True)}
 if face_ids is not None:kw['face_id']=np.ascontiguousarray(face_ids,dtype=np.uint64)
 mesh=mf.Mesh64(**kw);body=mf.Manifold(mesh)
 if body.status()!=mf.Error.NoError:raise RuntimeError('Manifold input '+str(body.status()))
 return body

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='envelope-001');ap.add_argument('--banks',default='43/16');ap.add_argument('--clearance-mm',type=float,default=.8);a=ap.parse_args()
 out=HERE/a.run;out.mkdir(exist_ok=False);t=time.monotonic();base=ROOT/'inputs/baseline-806.npz';z=np.load(base,allow_pickle=False);X=z['points_world_m'];F=z['source_O_faces'];m=len(F)
 V,Fclosed,boundary,loops=close_for_boolean(X,F[:,[0,2,1]])
 source=from_mesh(V,Fclosed,np.arange(len(Fclosed)));source_id=int(np.asarray(source.to_mesh64().run_original_id)[0]);print(json.dumps({'stage':'source','id':source_id,'vertices':source.num_vert(),'faces':source.num_tri(),'volume':source.volume(),'head_loops':[len(l) for l in loops]}),flush=True)
 helper=mod('envelope_banks',ROOT/'tools/human_skin_fat_bone_audit.py');T,ids,_,_,M,off,pos,_,_,pins=helper.read_bank_inputs(ROOT/'inputs/bank-export-039/source-preflight',ROOT/'tools')
 targets=[tuple(map(int,v.split('/'))) for v in a.banks.split(',')];offsets=[];entries=[]
 sphere=mf.Manifold.sphere(a.clearance_mm/1000,8)
 for body,bone in targets:
  tr=T[(ids[:,0]==body)&(ids[:,1]==bone)];P,ix=np.unique(tr.reshape(-1,3),axis=0,return_inverse=True);BF=ix.reshape(-1,3)
  BP=(P+off[body])@np.linalg.inv(M[body]).T+pos[body]
  b=from_mesh(BP,BF)
  envelope=b.minkowski_sum(sphere);envelope.num_tri();env= envelope.as_original();e_id=env.original_id();offsets.append(env)
  em=env.to_mesh64();np.savez_compressed(out/f'offset-{body}-{bone}.npz',points_world_m_f64=np.asarray(em.vert_properties)[:,:3],source_O_faces=np.asarray(em.tri_verts,dtype=np.int64),original_id=np.array([e_id]),body=np.array([body]),stable_bone_id=np.array([bone]))
  entry={'body':body,'stable_bone_id':bone,'input_triangles':len(BF),'envelope_triangles':env.num_tri(),'original_id':e_id,'offset_radius_mm':a.clearance_mm,'derivation':'Minkowski sum of exact registered bone with 8-segment inscribed sphere; geometric construction, not measured skin.'};entries.append(entry);print(json.dumps({'stage':'offset',**entry,'elapsed_s':time.monotonic()-t}),flush=True)
 # One connected result rather than independently translated regional patches.
 combined=mf.Manifold.batch_boolean([source]+offsets,mf.OpType.Add)
 if combined.status()!=mf.Error.NoError:raise RuntimeError(str(combined.status()))
 combined=combined.simplify(2e-7)
 mesh=combined.to_mesh64();PV=np.asarray(mesh.vert_properties)[:,:3];PF=np.asarray(mesh.tri_verts,dtype=np.int64);faceids=np.asarray(mesh.face_id,dtype=np.int64);runidx=np.asarray(mesh.run_index,dtype=np.int64);orig=np.asarray(mesh.run_original_id,dtype=np.int64)
 faceorig=np.empty(len(PF),dtype=np.int64)
 for i,oid in enumerate(orig):faceorig[runidx[i]//3:runidx[i+1]//3]=oid
 cap=(faceorig==source_id)&(faceids>=m);PF=PF[~cap];faceids=faceids[~cap];faceorig=faceorig[~cap]
 # Emit raw source convention (inward winding) used by the retained pipeline.
 PF=PF[:,[0,2,1]]
 Yunique,G,emission=emit(PV,PF,X)
 drop=np.zeros(len(G),dtype=bool)
 outpath=out/'candidate.npz'
 np.savez_compressed(outpath,points_world_m=Yunique,source_O_faces=G,envelope_source_original_id=faceorig,envelope_source_face_row=faceids,envelope_baseline_original_id=np.array([source_id]),envelope_baseline_sha256=np.array([sha(base)]),envelope_derivation_json=np.array([json.dumps(entries)]))
 # Exact bitwise original coordinates and triangles for source-derived output.
 lookup={tuple(row.view(np.uint32)):i for i,row in enumerate(X)};newtoold=np.array([lookup.get(tuple(row.view(np.uint32)),-1) for row in Yunique],dtype=np.int64)
 src=(faceorig==source_id);unchanged=0;sourcealtered=[]
 for i in np.flatnonzero(src):
  par=int(faceids[i]);old=newtoold[G[i]]
  if par<m and np.array_equal(np.sort(old),np.sort(F[par])):unchanged+=1
  else:sourcealtered.append(int(i))
 report={'host':'numi.local','baseline_sha256':sha(base),'script_sha256':sha(__file__),'candidate_sha256':sha(outpath),'source_original_id':source_id,'temporary_cap_triangles':len(Fclosed)-m,'removed_cap_triangles':int(cap.sum()),'original_head_boundary_edges':len(boundary),'source_head_loop_lengths':[len(l) for l in loops],'output_vertices':len(Yunique),'output_faces':len(G),'exact_float32_collapsed_faces':int(drop.sum()),'float32_emission':emission,'bitwise_unchanged_source_triangles':unchanged,'source_retriangulated_face_rows':sourcealtered,'derived_envelope_faces':int(np.count_nonzero(~src)),'offsets':entries,'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'limitations':['Topology reconstruction only; layer volume, conservative mass and native mechanics are not yet compiled.','Temporary head caps were removed; original head boundary remains open and is audited separately.'],'elapsed_s':time.monotonic()-t}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_retriangulated_face_rows','offsets']}),flush=True)
if __name__=='__main__':main()
