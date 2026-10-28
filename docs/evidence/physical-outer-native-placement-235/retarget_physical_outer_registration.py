from pathlib import Path
import csv,json,hashlib,shutil,subprocess
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
HERE=Path(__file__).resolve().parent;OUT=HERE/'registered-surfaces-002'
BASE=P/'volume-worker/registered-surface-sidecar-222/attempt-018-native-current-reference'
CAN=P/'volume-worker/registered-surface-sidecar-222/attempt-016-kindaware-registration'
CAP=P/'native-parent-label-capture-016/compiled-parent-cell-mass.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(newline='') as f:
  r=csv.DictReader(f);return list(r.fieldnames),list(r)
def write(p,h,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=h,lineterminator='\n');w.writeheader();w.writerows(rows)
def fmt(v):return format(float(v),'.17g')
report=json.loads((HERE/'report.json').read_text());assert report['exact_self_pairs']==0 and report['degenerate']==report['flipped']==0
mesh=HERE/'O-direct-native-placement.npz';assert sha(mesh)==report['output_sha256']
z=np.load(mesh,allow_pickle=False);X=np.asarray(z['points_world_m'],np.float32).astype(float);F=z['source_O_faces'].astype(np.int64)
R=np.asarray(report['placement']['rotation_row_major']).reshape(3,3);T=np.asarray(report['placement']['translation_m'])
shutil.copytree(BASE,OUT)
h,rows=read(BASE/'registered-O.csv');_,canonical=read(CAN/'registered-O.csv');_,faces=read(BASE/'registered-O-faces.csv')
assert len(X)==len(rows)==len(canonical)==32002
source_ids=np.array([int(r['O033_source_vertex_row']) for r in rows]);zsource=z['candidate_vertex_to_O033_row'].astype(np.int64)
assert len(set(source_ids))==len(set(zsource))==len(X) and set(source_ids)==set(zsource)
source_to_z={int(v):i for i,v in enumerate(zsource)};source_to_side={int(v):i for i,v in enumerate(source_ids)}
side_to_z=np.array([source_to_z[int(v)] for v in source_ids]);z_to_side=np.array([source_to_side[int(v)] for v in zsource])
assert np.array_equal(z_to_side[F],np.array([[int(r[k]) for k in ('v0','v1','v2')] for r in faces]))
X=X[side_to_z]
_,fr=read(BASE/'donor-frames.csv');frames={int(r['donor']):r for r in fr}
_,pn=read(str(CAP)+'.nodes.csv');parent=np.array([[float(r['current_'+a+'_m']) for a in 'xyz'] for r in pn]);_,pt=read(str(CAP)+'.tets.csv');tets=np.array([[int(r['parent_node'+str(c)]) for c in range(4)] for r in pt])
rh,regs=read(BASE/'node-parent-registration.csv');regby={(r['layer'],int(r['surface_vertex_row'])):r for r in regs};delta=[]
for i,(row,can) in enumerate(zip(rows,canonical)):
 assert int(row['vertex'])==i==int(can['vertex']);point=X[i];oldpoint=np.array([float(row[k]) for k in ('wx','wy','wz')]);delta.append(np.linalg.norm(point-oldpoint))
 donor=int(row['donor']);frame=frames[donor];origin=np.array([float(frame['current_'+a]) for a in 'xyz']);M=np.array([[float(frame['current_R'+str(j)+str(a)]) for j in range(3)] for a in range(3)])
 predicted=R@np.array([float(can[k]) for k in ('wx','wy','wz')])+T
 sr=np.array([float(row['source_residual_'+a+'_m']) for a in 'xyz'])+point-predicted
 normal=R@np.array([float(can[k]) for k in ('nx','ny','nz')]);normal/=np.linalg.norm(normal);local=M@(point-origin)
 for k,v in zip(('wx','wy','wz'),point):row[k]=fmt(v)
 for k,v in zip(('nx','ny','nz'),normal):row[k]=fmt(v)
 for k,v in zip(('lx','ly','lz'),local):row[k]=fmt(v)
 for a in range(3):
  for b in range(3):row['m'+str(a)+str(b)]=fmt(M[a,b])
 for a,v in zip('xyz',sr):row['residual_'+a]=row['source_residual_'+a+'_m']=fmt(v)
 reg=regby['O',i];tet=int(reg['parent_tet_row']);assert 0<=tet<len(tets);bary=np.array([float(reg['parent_bary'+str(k)]) for k in range(4)])
 residual=point-bary@parent[tets[tet]];distance=np.linalg.norm(residual);assert distance<=.02
 for a,v in zip('xyz',residual):reg['parent_residual_'+a+'_m']=fmt(v)
 for a,v in zip('xyz',sr):reg['source_residual_'+a+'_m']=fmt(v)
 reg['parent_registration_distance_m']=fmt(distance)
write(OUT/'registered-O.csv',h,rows);write(OUT/'node-parent-registration.csv',rh,regs)
unchanged=[q.name for q in BASE.iterdir() if q.is_file() and q.name not in ('registered-O.csv','node-parent-registration.csv','receipt.json')]
assert all(sha(BASE/name)==sha(OUT/name) for name in unchanged)
receipt={'scope':'same-topology physical O native placement registration; no changed volume or physical admission','input_surface_packet':str(BASE),'input_surface_packet_receipt_sha256':sha(BASE/'receipt.json'),'input_outer_npz':str(mesh),'input_outer_sha256':sha(mesh),'exact_outer_self_report_sha256':sha(HERE/'report.json'),'source_ancestry_preserved':True,'parent_barycentric_and_semantics_preserved':True,'changed_registered_O_rows':len(rows),'max_O_difference_from_LBS_observation_m':float(max(delta)),'unchanged_companion_files':unchanged,'native_steps':0,'admitted':False,'outputs_sha256':{q.name:sha(q) for q in OUT.iterdir() if q.is_file() and q.name!='receipt.json'}}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
cmd=[str(P/'P/native-current-parent-registration-229/check'),str(OUT),str(CAP),'--expect-pass'];result=subprocess.run(cmd,text=True,capture_output=True);(HERE/'registration-check.stdout.txt').write_text(result.stdout);(HERE/'registration-check.stderr.txt').write_text(result.stderr)
check={'command':cmd,'exit_code':result.returncode,'binary_sha256':sha(cmd[0]),'input_packet_receipt_sha256':sha(OUT/'receipt.json'),'parent_packet_sha256':sha(CAP),'physical_steps':0,'changed_volume_admitted':False};(HERE/'registration-check.json').write_text(json.dumps(check,indent=2)+'\n');print(result.stdout,flush=True);print(json.dumps(check),flush=True);raise SystemExit(result.returncode)
