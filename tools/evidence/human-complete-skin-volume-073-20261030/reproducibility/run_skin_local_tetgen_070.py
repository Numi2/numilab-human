#!/usr/bin/env python3
"""Batch four remaining restored Skin cavities using the corrected mesher."""
from pathlib import Path
from fractions import Fraction
import numpy as np,json,sys,importlib.util,hashlib,subprocess,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
D=H/'skin010-local-tetgen-070';D.mkdir(exist_ok=False)
with np.load(H/'skin010-inverse-boundary-splits-068/skin-boundary-restoration-proposal.npz')as z:old={k:z[k]for k in z.files}
with np.load(H/'skin010-restored-cavity-batch-069/skin-restored-boundary-proposal.npz')as z:a={k:z[k]for k in z.files}
p=a['points_world_m'];t=a['tetrahedra'];mat=a['material_indices'];oldp=old['points_world_m'];node_map=a['source068_node_to_candidate_node'];cellmap={int(r):i for i,r in enumerate(a['retained_source068_cell_rows'])};pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
owner=P/'deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py';sys.path.insert(0,str(owner.parent.parent));exact=__import__('numilab_human.cardiac_cavity_intersections',fromlist=['_audit_pair'])
domain=H/'native-domain-217-source-004/build_native_domain_217.py';spec=importlib.util.spec_from_file_location('source004_skin070',domain);dom=importlib.util.module_from_spec(spec);sys.modules[spec.name]=dom;spec.loader.exec_module(dom)
binary=H/'attribute-propagation-062/hull-attribute-fix-001/tetgen-hull-attribute-fix';assert hashlib.sha256(binary.read_bytes()).hexdigest()=='01d15d76f8dfcdcd84f10fac280f7aeed3de1ae24991b0a359e05006c99e6bcb'
source_report=json.loads((H/'skin010-restored-cavity-batch-069/report.json').read_text());details=[];newpoints=[];newtets=[];removed=[];newmats=[]
def key(f):return tuple(sorted(map(int,f)))
def cyc(f):
 a,b,c=map(int,f);return min((a,b,c),(b,c,a),(c,a,b))
for item in source_report['details']:
 if item['status'].startswith('PASS'):continue
 ci=item['component'];cd=D/f'component-{ci:03d}';cd.mkdir()
 with np.load(H/f'skin010-restored-cavity-batch-069/component-{ci:03d}/restored-boundary.npz')as z:b0=z['boundary_faces_global_node_rows'];ids0=z['source_cell_rows']
 assert np.all(node_map[b0]>=0);b=node_map[b0];ids=np.array([cellmap[int(i)]for i in ids0],np.int64)
 used=np.unique(b);lookup={int(v):j for j,v in enumerate(used)};lp=p[used];lf=np.array([[lookup[int(v)]for v in face]for face in b])
 rec,_=exact.float32_surface_records(lp,lf);gate=exact._audit_pair(rec,rec,same_surface=True)
 rr={'component':ci,'source_cells':len(ids),'fixed_points':len(lp),'restored_faces':len(lf),'exact_boundary_self_intersections':gate}
 if gate['count']:rr['status']='REJECT_SELF_INTERSECTING_RESTORED_CUT';details.append(rr);continue
 prepared=exact.prepare_float32_point_location_meters(lp,lf);q=p[t[ids]].astype(np.float64);dv=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0]);seed=None
 for k in np.argsort(-np.abs(dv)):
  s=tuple(sum(Fraction(float(v))for v in p[t[ids[k]],axis])/4 for axis in range(3))
  location=exact.point_location_prepared_indexed(s,prepared)
  if location['location']=='inside':seed=np.array([float(x)for x in s]);break
 if seed is None:rr['status']='REJECT_NO_CERTIFIED_SEED';details.append(rr);continue
 edges=np.concatenate([lf[:,[0,1]],lf[:,[1,2]],lf[:,[2,0]]]);minedge=float(np.linalg.norm(lp[edges[:,0]].astype(np.float64)-lp[edges[:,1]].astype(np.float64),axis=1).min());diag=float(np.linalg.norm(lp.max(0).astype(np.float64)-lp.min(0).astype(np.float64)));tol=min(1e-12,.1*minedge/diag)
 poly=cd/'skin.poly';dom.write_poly(poly,lp,lf,np.ones(len(lf),np.int32),[],[(seed,15)])
 cmd=[str(binary),f'-pYYAq1.2/10S128O0/0/0T{tol:.17g}',str(poly)];tic=time.monotonic();pr=subprocess.run(cmd,capture_output=True,text=True,timeout=12);rr.update(argv=cmd,exit_code=pr.returncode,wall_s=time.monotonic()-tic,seed_world_m=seed.tolist());(cd/'stdout.log').write_text(pr.stdout);(cd/'stderr.log').write_text(pr.stderr)
 if pr.returncode:rr['status']='TETGEN_FAILED';details.append(rr);continue
 nd=np.loadtxt(cd/'skin.1.node',skiprows=1);el=np.loadtxt(cd/'skin.1.ele',skiprows=1,dtype=np.int64)
 op=nd[:,1:4].astype(np.float32);ot=el[:,1:5]-1
 assert np.array_equal(nd[:,0],np.arange(1,len(nd)+1))
 if not np.array_equal(op[:len(lp)],lp):rr['status']='REJECT_FIXED_NODES_CHANGED';details.append(rr);continue
 if len(np.unique(op.view(np.uint32),axis=0))!=len(op):rr['status']='REJECT_F32_DUPLICATE_NEW_POINTS';details.append(rr);continue
 q=op[ot].astype(np.float64);v=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0]);rr['F32_nonpositive_cells']=int(np.sum(v<=0))
 if np.any(v<=0)or np.any(el[:,5]!=15):rr['status']='REJECT_NONPOSITIVE_OR_MATERIAL';details.append(rr);continue
 ff=ot[:,pat].reshape(-1,3);_,ix,ct=np.unique(np.sort(ff,axis=1),axis=0,return_index=True,return_counts=True);ob=ff[ix[ct==1]]
 if np.any(ct>2)or sorted(cyc(x)for x in ob)!=sorted(cyc(x)for x in lf):rr['status']='REJECT_EXACT_ORIENTED_BOUNDARY';details.append(rr);continue
 records,_=exact.float32_surface_records(op,ff[ix]);cg=exact._audit_pair(records,records,same_surface=True)
 if cg['count']:rr.update(status='REJECT_OUTPUT_INTERSECTIONS',gate=cg);details.append(rr);continue
 omap=np.r_[used,np.arange(len(p)+len(newpoints),len(p)+len(newpoints)+len(op)-len(lp))]
 newpoints.extend(op[len(lp):]);newtets.append(omap[ot]);newmats.extend(el[:,5].tolist());removed.extend(ids.tolist());rr.update(status='PASS_LOCAL_RESTORED_CAVITY',new_cells=len(ot),new_points=len(op)-len(lp));details.append(rr)
mask=np.ones(len(t),bool);mask[removed]=False;pp=np.vstack([p,np.asarray(newpoints,np.float32).reshape(-1,3)]);tt=np.vstack([t[mask],*newtets]);mm=np.r_[mat[mask],newmats].astype(np.int32)
used=np.unique(tt);assert set(range(202037))<=set(map(int,used));rows=np.r_[np.arange(202037),used[used>=202037]];nm=np.full(len(pp),-1,np.int64);nm[rows]=np.arange(len(rows));pp=pp[rows];tt=nm[tt]
q=pp[tt].astype(np.float64);dv=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0]);ff=tt[:,pat].reshape(-1,3);_,ix,ct=np.unique(np.sort(ff,axis=1),axis=0,return_index=True,return_counts=True);bb=ff[ix[ct==1]]
with np.load(H/'verified-volume-preparation-010-merged038005/plcs/skin-shell.npz')as z:sf=z['faces']
bk={key(x)for x in bb};sk={key(x)for x in sf}
r={'status':'SKIN_RESTORED_LOCAL_TETGEN_BATCH_PARTIAL','physical_admission':False,'runtime':0,'accepted_steps':0,'F32_zero':int(np.sum(dv==0)),'F32_negative':int(np.sum(dv<0)),'missing_original_PLC_facets':len(sk-bk),'extra_boundary_facets':len(bk-sk),'candidate_cells':len(tt),'candidate_nodes':len(pp),'nonmanifold_tet_faces':int(np.sum(ct>2)),'details':details}
np.savez_compressed(D/'skin-local-tetgen-proposal.npz',points_world_m=pp,tetrahedra=tt,material_indices=mm,boundary_faces=bb,source069_node_to_candidate_node=nm[:len(p)],retained_source069_cell_rows=np.flatnonzero(mask))
(D/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
