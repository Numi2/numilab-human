from pathlib import Path
import numpy as np,json,hashlib,time,importlib.util,sys
from scipy.spatial import cKDTree
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706';out=H/'tetgen-material-ownership-diagnostic-055'
src=H/'tetgen-explicit-no-improvement-053/skin037-run-001';t0=time.monotonic()
def read(p):
 with p.open()as f:f.readline();return np.loadtxt(f,comments='#')
n=read(src/'skin.1.node');e=read(src/'skin.1.ele');p=n[:,1:4];pf=p.astype(np.float32).astype(np.float64);t=e[:,1:5].astype(np.int64)-1
q=pf[t];v=np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6
z=np.load(H/'geometry-feasibility-008-warning037/skin-shell.npz');shell={k:z[k]for k in z.files};sp=shell['points_world_m'];sf=shell['faces']
assert np.array_equal(pf[:len(sp)],sp.astype(np.float64))
r37=json.loads((H/'quality-warning-connected-union-037/run-002/report.json').read_text());r38=json.loads((H/'quality-warning-connected-union-038/run-004/report.json').read_text())
roles={}
for role in ('O','I'):
 zs=np.load(r37['outputs'][role]['path']);d={k:zs[k]for k in zs.files};zc=np.load(r38['outputs'][role]['path']);dc={k:zc[k]for k in zc.files}
 pp=np.asarray(d['points_world_m'],dtype=np.float32);ff=np.asarray(d['faces'],dtype=np.int64)
 rows=shell['registered_'+role+'_shell_node_rows']; assert np.array_equal(sp[rows],pp)
 inv=np.asarray(dc['point_parent_input_to_output_row'],dtype=np.int64)
 incidence=[[]for _ in pp]
 for fi,tri in enumerate(ff):
  for vi in tri:incidence[int(vi)].append(fi)
 roles[role]=(pp,ff,rows,inv,incidence,cKDTree(pp))
inverse={}
for role,(_,_,rows,*rest)in roles.items():
 for ri,si in enumerate(rows):inverse.setdefault(int(si),[]).append((role,ri))
records=[];role_seeds={r:set()for r in roles}
for ci in np.flatnonzero((e[:,5]!=15)|(v<=0)):
 anchors={r:set()for r in roles};direct={r:set()for r in roles}
 for ni in t[ci]:
  if int(ni)in inverse:
   for role,ri in inverse[int(ni)]:anchors[role].add(ri);direct[role].add(ri)
  else:
   # Explicit diagnostic nearest nodes only, never ancestry or geometry ownership.
   for role,(pp,ff,rows,inv,inc,tree)in roles.items():
    dist,ri=tree.query(pf[int(ni)])
    if dist<=1e-5:anchors[role].add(int(ri))
 rs={}
 for role,(pp,ff,rows,inv,inc,tree)in roles.items():
  faces=sorted({fi for ri in anchors[role]for fi in inc[ri]})
  role_seeds[role].update(faces)
  rs[role]={'source037_direct_vertex_rows':sorted(direct[role]),'source037_near_vertex_rows':sorted(anchors[role]-direct[role]),'source037_incident_face_rows':faces,'current038_vertex_rows':[int(inv[ri])for ri in sorted(anchors[role])if inv[ri]>=0]}
 records.append({'element_row':int(ci),'attribute':float(e[ci,5]),'signed_volume_f32_m3':float(v[ci]),'node_rows':t[ci].tolist(),'steiner_node_rows':[int(x)for x in t[ci]if x>=len(sp)],'centroid_world_m':pf[t[ci]].mean(0).tolist(),'roles':rs})
report={'status':'EXPLICIT_CELL_WITNESSES_DIAGNOSTIC_NOT_GEOMETRIC_ANCESTRY','source_result_sha256':hashlib.sha256((out/'report.json').read_bytes()).hexdigest(),'cells':records,'source037_face_seeds':{r:sorted(s)for r,s in role_seeds.items()},'role_source_paths':{r:r37['outputs'][r]['path']for r in roles},'role_source_sha256':{r:r37['outputs'][r]['sha256']for r in roles},'physical_admission':False,'wall_s':time.monotonic()-t0}
path=out/'cell-role-witnesses-001.json';path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cells':len(records),'face_seeds':{r:len(s)for r,s in role_seeds.items()},'input_corner_cells':sum(any(rec['roles'][r]['source037_direct_vertex_rows']for r in roles)for rec in records),'steiner_cell_count':sum(bool(rec['steiner_node_rows'])for rec in records),'wall_s':report['wall_s']}))

