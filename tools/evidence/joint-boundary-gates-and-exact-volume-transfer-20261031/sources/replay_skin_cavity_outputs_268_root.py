#!/usr/bin/env python3
"""Recheck retained TetGen cavity outputs through existing exact owners and compose.
Native quality debt is an explicit bounded construction state; runtime admission stays false.
"""
from pathlib import Path
import numpy as np,json,hashlib,argparse,time,sys,importlib.util
H=Path(__file__).resolve().parent
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
M=module('cavity247for268',H/'remesh_current_surface_cavities_247_root.py')
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def run(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False);root=H/'native-apex-skin-target-265-root/cavities-247-001';rp=root/'report.json';r=json.loads(rp.read_text())
 src=Path(r['input']);assert sha(src)==r['input_sha256']=='c9fc3f627083b07015a4cfa36c048c8e60f8997b7b7528f03645272cd98802a3'
 with np.load(src) as z:v={k:z[k] for k in z.files}
 p=v['points_world_m'];t=v['tetrahedra'];mat=v['material_indices']
 with np.load(root/'connected-cavity-shells.npz') as z:cv={k:z[k] for k in z.files}
 owner=H.parent/'deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py';sys.path.insert(0,str(owner.parent.parent));E=__import__('numilab_human.cardiac_cavity_intersections',fromlist=['_audit_pair']);assert sha(Path(E.__file__))=='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
 approved=[];failed=[];allparents=[];debt=0;delta=0
 for ci,rr in enumerate(r['components']):
  case=root/f'component-{ci:04d}'
  if not (case/'cavity.1.node').exists() or not (case/'cavity.1.ele').exists():
   failed.append({'component':ci,'reason':'NO_FROZEN_TETGEN_OUTPUT','original_status':rr['status'],'original_error':rr.get('error')});continue
  try:
   co=cv['cell_offsets'];po=cv['point_offsets'];fo=cv['face_offsets'];rows=cv['input_cell_rows'][co[ci]:co[ci+1]];grows=cv['global_point_rows'][po[ci]:po[ci+1]];lf=cv['local_faces'][fo[ci]:fo[ci+1]]
   fixed=p[grows];assert M.closed(lf)==(True,0);top=M.manifold_topology(lf);assert top['boundary_components']==1 and not top['bad_vertex_links']
   nn=M.readtable(case/'cavity.1.node');ee=M.readtable(case/'cavity.1.ele');op=nn[:,1:4].astype(np.float32);nid=nn[:,0].astype(np.int64);ix=np.searchsorted(nid,ee[:,1:5].astype(np.int64));assert np.array_equal(nid[ix],ee[:,1:5].astype(np.int64));ot=ix.astype(np.int64)
   assert len(np.unique(mat[rows]))==1 and ee.shape[1]>=6 and np.all(ee[:,5]==mat[rows[0]])
   lookup={M.m.key(x):i for i,x in enumerate(fixed)};assert len(lookup)==len(fixed)
   omap=np.asarray([lookup.get(M.m.key(x),-1) for x in op],np.int64);assert np.all(omap>=0),'unexpected interior Steiner point in S0 replay'
   assert len(np.unique(omap))==len(omap);assert np.array_equal(op,fixed[omap])
   bd=M.m.boundary(ot);assert np.array_equal(M.m.boundary_identity(omap[bd]),M.m.boundary_identity(lf))
   nd,nq=M.m.m.measurements(op,ot);od,oq=M.m.m.measurements(p,t[rows]);assert np.all(nd>0) and np.all(od>0)
   assert int(sum(~M.m.m.passes(od,oq)))==rr['old_bad']
   ip=M.m.m.integer_points(op);det=[M.m.m.exact_det(ip,c) for c in ot];shell=M.exact_volume(fixed,lf);assert min(det)>0 and sum(det)==shell
   fi=ot[:,M.m.m.PAT].reshape(-1,3);_,ui=np.unique(np.sort(fi,axis=1),axis=0,return_index=True);records,_=E.float32_surface_records(op,fi[ui]);gate=E._audit_pair(records,records,same_surface=True);assert gate['count']==0
   srecords,_=E.float32_surface_records(fixed,lf);sg=E._audit_pair(srecords,srecords,same_surface=True);assert sg['count']==0
   newbad=int(sum(~M.m.m.passes(nd,nq)));qdebt=max(0,newbad-int(rr['old_bad']));assert debt+qdebt<=a.maximum_total_construction_debt,'explicit construction debt budget exceeded'
   oldip=M.m.m.integer_points(p[np.unique(t[rows])]);used=np.unique(t[rows]);imap={int(v):i for i,v in enumerate(used)};oldsum=sum(M.m.m.exact_det(oldip,[imap[int(v)] for v in c]) for c in t[rows])
   gt=grows[omap[ot]];assert len(np.unique(np.sort(gt,axis=1),axis=0))==len(gt)
   opath=a.output/f'operation-{ci:04d}.npz';np.savez_compressed(opath,tetrahedra=gt,parent_cell_rows=rows,target_shell_faces_global=grows[lf])
   record={'component':ci,'status':'PASS_EXACT_FROZEN_OUTPUT_REPLAY','parent_cell_rows':rows.tolist(),'source_cells':len(rows),'new_cells':len(gt),'old_native_quality_failures':rr['old_bad'],'new_native_quality_failures':newbad,'construction_quality_debt':qdebt,'exact_internal_face_pairs':0,'exact_shell_self_pairs':0,'exact_target_shell_volume':True,'source_to_target_volume_delta_determinant':str(shell-oldsum),'source_to_target_volume_delta_m3':(shell-oldsum)/(6*2**447),'replacement':str(opath),'replacement_sha256':sha(opath),'frozen_node_sha256':sha(case/'cavity.1.node'),'frozen_ele_sha256':sha(case/'cavity.1.ele'),'lineage_semantics':'full consumed source-cell patch union, not per-cell overlap'}
   approved.append((rows,gt,int(mat[rows[0]]),record));allparents.extend(rows.tolist());debt+=qdebt;delta+=shell-oldsum
  except Exception as e:failed.append({'component':ci,'reason':repr(e)})
 assert len(np.unique(allparents))==len(allparents);keep=np.ones(len(t),bool);keep[allparents]=False;nt=np.vstack([t[keep],*(x[1] for x in approved)]);nm=np.concatenate([mat[keep],*(np.full(len(x[1]),x[2],np.int32) for x in approved)]);bd=M.m.boundary(nt);nd,nq=M.m.m.measurements(p,nt);assert np.all(nd>0)
 desired=v['desired_boundary_faces'];dk=M.sortedkeys(desired);bk=M.sortedkeys(bd);missing=dk-bk;extra=bk-dk
 out=a.output/'skin-frozen-cavity-replay-candidate.npz';np.savez_compressed(out,points_world_m=p,tetrahedra=nt,material_indices=nm,boundary_faces=bd,desired_boundary_faces=desired,output_cell_to_input_cell_rows=np.r_[np.flatnonzero(keep),np.full(len(nt)-sum(keep),-1,np.int64)],source_input_sha256=np.asarray([sha(src)]),**{k:v[k] for k in ('solver_infeasible_cavity_sha256','kernel_fan_no_progress_sha256') if k in v})
 report={'status':'PASS_FROZEN_CAVITY_EXACT_REPLAY_CONSTRUCTION_ONLY','physical_admission':False,'native_steps':0,'source_sha256':sha(Path(__file__)),'input_report':str(rp),'input_report_sha256':sha(rp),'input':str(src),'input_sha256':sha(src),'candidate':str(out),'candidate_sha256':sha(out),'original_point_array_unchanged':True,'passing_components':len(approved),'failed_components':failed,'operations':[x[3] for x in approved],'temporary_construction_debt':debt,'temporary_construction_debt_limit':a.maximum_total_construction_debt,'native_quality_final_gate_unchanged':True,'remaining_nonpositive_cells':int(sum(nd<=0)),'remaining_native_quality_failures':int(sum(~M.m.m.passes(nd,nq))),'desired_missing_faces':len(missing),'current_extra_faces':len(extra),'source_to_target_volume_delta_determinant':str(delta),'source_to_target_volume_delta_m3':delta/(6*2**447),'complete_physical_mass_mapping':'PENDING','elapsed_s':time.monotonic()-tic}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='operations'},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--maximum-total-construction-debt',type=int,default=500);run(ap.parse_args())
