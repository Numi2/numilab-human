#!/usr/bin/env python3
"""Restore original PLC edge facets by valid inverse construction-edge splits.
Every contraction is fail-closed on link and exact positive incident-cell gates.
This is mesh preparation, before physical mass/state allocation.
"""
from pathlib import Path
import numpy as np,json,itertools,hashlib
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
S=H/'skin010-prepared-current-volume-068/skin-current-volume.npz';D=H/'skin010-inverse-boundary-splits-068';D.mkdir(exist_ok=False)
with np.load(S)as z:a={k:z[k]for k in z.files}
p=a['points_world_m'];t=a['tetrahedra'].copy();materials=a['material_indices'];mp=a['source_candidate_node_to_quotient_node'];original=202037;keep=np.ones(len(t),bool);identity=np.arange(len(p))
diagnostic=H/'tetgen-scale-relative-tolerance-059/boundary-subdivision-audit-001/nearest-edge-diagnostic.json';dr=json.loads(diagnostic.read_text())['skin']['rows']
parents={}
for r in dr:
 row=int(mp[r['output_node_row']])
 if row>=original:
  if row in parents:assert parents[row]['closest_missing_original_edge']==r['closest_missing_original_edge']
  parents[row]=r
def link(cells,vertices):
 vertices=set(vertices);out=set()
 for cell in cells:
  if not vertices<=set(map(int,cell)):continue
  rest=sorted(set(map(int,cell))-vertices)
  for k in range(len(rest)+1):out.update(itertools.combinations(rest,k))
 return out
cache={}
def ip(row):
 row=int(row)
 if row not in cache:
  cache[row]=tuple(int(float(v).as_integer_ratio()[0])<<(149-(float(v).as_integer_ratio()[1].bit_length()-1))for v in p[row])
 return cache[row]
def det(tet):
 a,b,c,d=map(ip,tet);u=[b[k]-a[k]for k in range(3)];v=[c[k]-a[k]for k in range(3)];w=[d[k]-a[k]for k in range(3)]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])
accepted=[];attempts=[];pending=set(parents)
for sweep in range(4):
 progress=0
 for m in sorted(pending):
  incident=np.flatnonzero(keep&np.any(t==m,axis=1))
  if len(incident)==0:pending.remove(m);continue
  rec=parents[m];choices=[]
  for u in rec['closest_missing_original_edge']:
   u=int(u);local=t[keep&np.any((t==m)|(t==u),axis=1)]
   lu=link(local,[u]);lm=link(local,[m]);le=link(local,[u,m]);ok=(lu&lm)==le
   rr={'sweep':sweep,'construction_node_row':m,'source_endpoint_row':u,'incident_cells':len(incident),'link_condition':ok}
   if not ok:rr['status']='REJECT_LINK';attempts.append(rr);continue
   trial=t[incident].copy();trial[trial==m]=u;repeated=np.any(np.diff(np.sort(trial,axis=1),axis=1)==0,axis=1)
   nonrep=trial[~repeated];dv=[det(x)for x in nonrep]
   if not dv or any(x<=0 for x in dv):
    rr.update(status='REJECT_NONPOSITIVE_RESULT',nonpositive_cells=sum(x<=0 for x in dv));attempts.append(rr);continue
   old=[det(x)for x in t[incident]]
   rr.update(status='PASS_LOCAL_INVERSE_EDGE_SPLIT',removed_cell_rows=incident[repeated].tolist(),removed_cells=len(incident[repeated]),remaining_incident_cells=len(nonrep),old_negative_incident_cells=sum(x<0 for x in old),old_zero_incident_cells=sum(x==0 for x in old),exact_old_signed_det6_lattice_sum=str(sum(old)),exact_new_signed_det6_lattice_sum=str(sum(dv)),original_edge=rec['closest_missing_original_edge'],parent_original_PLC_face_rows=rec['incident_missing_input_face_rows'],source_edge_mapping_F64_distance_m=rec['distance_m'])
   choices.append((min(dv),u,trial,repeated,rr))
  if not choices:continue
  _,u,trial,repeated,rr=max(choices,key=lambda x:(x[0],-x[1]))
  t[incident]=trial;keep[incident[repeated]]=False;identity[identity==m]=u
  accepted.append(rr);pending.remove(m);progress+=1
 if not progress:break
t=t[keep];mat=materials[keep];used=np.unique(t);assert np.array_equal(np.intersect1d(used,np.arange(original)),np.arange(original))
rows=np.r_[np.arange(original),used[used>=original]];compact=np.full(len(p),-1,np.int64);compact[rows]=np.arange(len(rows));pp=p[rows];tt=compact[t]
pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]]);faces=tt[:,pat].reshape(-1,3);sf=np.sort(faces,axis=1);_,first,ct=np.unique(sf,axis=0,return_index=True,return_counts=True);b=faces[first[ct==1]]
edges=np.sort(np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]),axis=1);_,ec=np.unique(edges,axis=0,return_counts=True)
q=pp[tt].astype(np.float64);dd=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])
with np.load(H/'verified-volume-preparation-010-merged038005/plcs/skin-shell.npz')as z:source_faces=z['faces'];source_volume=float(z['shell_signed_volume_m3'])
def pack(f):
 f=np.sort(f.astype(np.uint64),axis=1);return(f[:,0]<<40)|(f[:,1]<<20)|f[:,2]
missing=np.flatnonzero(~np.isin(pack(source_faces),pack(b)));extra=np.flatnonzero(~np.isin(pack(b),pack(source_faces)))
report={'status':'BOUNDED_INVERSE_CONSTRUCTION_EDGE_SPLITS_PARTIAL_NOT_PHYSICAL_ADMISSION','physical_admission':False,'runtime':0,'accepted_steps':0,'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'source_cells':len(a['tetrahedra']),'candidate_cells':len(tt),'source_nodes':len(p),'candidate_nodes':len(pp),'original_source_nodes_preserved':bool(np.array_equal(pp[:original],p[:original])),'accepted_edge_splits':len(accepted),'remaining_boundary_construction_nodes':len(pending),'F32_zero_cells':int(np.sum(dd==0)),'F32_negative_cells':int(np.sum(dd<0)),'nonmanifold_tet_faces':int(np.sum(ct>2)),'boundary_edges_not_two':int(np.sum(ec!=2)),'duplicate_cells':len(tt)-len(np.unique(np.sort(tt,axis=1),axis=0)),'missing_original_PLC_facets':len(missing),'extra_boundary_facets':len(extra),'source_closed_shell_volume_m3':source_volume,'candidate_oriented_tet_volume_m3':float(dd.sum()/6),'signed_volume_difference_from_original_PLC_m3':float(dd.sum()/6-source_volume),'accepted':accepted,'rejected_attempts':attempts,'pending_construction_nodes_source065_rows':sorted(pending)}
np.savez_compressed(D/'skin-boundary-restoration-proposal.npz',points_world_m=pp,tetrahedra=tt,material_indices=mat,boundary_faces=b,source065_node_to_candidate_node=compact[identity],retained_source065_cell_rows=np.flatnonzero(keep),missing_original_PLC_face_rows=missing,extra_candidate_boundary_face_rows=extra)
(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in('accepted','rejected_attempts','pending_construction_nodes_source065_rows')}))
