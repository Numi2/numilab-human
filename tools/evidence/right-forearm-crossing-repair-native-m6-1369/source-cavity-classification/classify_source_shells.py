from __future__ import annotations
import hashlib,json,pickle,sys,resource,importlib.util
from pathlib import Path
from fractions import Fraction as F
import numpy as np
ROOT=Path('/Users/n/numi-human-fds132-m6-1350')
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
PUB=Path('/Users/n/numi-human-left-pq-publication-1344/src')
SOURCE=A/'right-pq-native-mtp-two-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
MANIFEST=A/'right-pq-native-mtp-two-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'
CAND=ROOT/'attempt-177-right-pentagon-option2-checkpoint-001/result-002/candidate.pkl'
OUT=ROOT/'attempt-181-right133-inherited-shell-classification-001/result-001'
OUT.mkdir(parents=True,exist_ok=False)
try:
 lim=int(2.5*1024**3);resource.setrlimit(resource.RLIMIT_AS,(lim,lim))
except Exception:pass
sys.path.insert(0,str(PUB))
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
LOADER=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
sp=importlib.util.spec_from_file_location('forward_all_positive_weights',LOADER);fw=importlib.util.module_from_spec(sp);sys.modules[sp.name]=fw;sp.loader.exec_module(fw)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)=='9d27e8060dce97d8b3d0f17ff945aeed0c83158341b95721cb27000269011967'
assert sha(CAND)=='2be23c705be2a6589a709481f1dcf57a09b87481e6524909ea95c6048818e5e9'
manifest=json.loads(MANIFEST.read_text());assert manifest['payload']['sha256']==sha(SOURCE)
tissue=fw.load_tissue(SOURCE);raw=fw.row_data(tissue,133)
source_points=[tuple(F.from_float(float(x)) for x in p) for p in raw['positions']]
source_faces=np.asarray(raw['faces'],dtype=np.int64)
source_top=analyze_topology(raw['positions'].tolist(),source_faces.tolist())
source_comps=[list(map(int,c)) for c in source_top['face_components']]

def signed_volume6(points,faces):
 v=F(0)
 for f in faces:
  a,b,c=(points[int(i)] for i in f)
  cr=(b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0])
  v+=sum(a[i]*cr[i] for i in range(3))
 return v

def component_records(points,faces,face_ids):
 f=[tuple(map(int,faces[i])) for i in face_ids]
 r=ci._records(points,f)
 return r

def comp_summary(points,faces,comps):
 out=[];prepared=[];reps=[]
 for ci0,face_ids in enumerate(comps):
  tris=[tuple(map(int,faces[i])) for i in face_ids]
  vol6=signed_volume6(points,tris)
  rec=component_records(points,faces,face_ids)
  topo=ci._validate_closed_oriented_record_topology(rec)
  first=tris[0];rep=tuple(sum(points[v][axis] for v in first)/3 for axis in range(3))
  used=sorted({int(v) for f in tris for v in f})
  lo=[min(points[v][a] for v in used) for a in range(3)];hi=[max(points[v][a] for v in used) for a in range(3)]
  out.append({'component_index':ci0,'face_count':len(tris),'used_vertex_count':len(used),'signed_volume_m3':float(vol6/6),'signed_volume6_exact':str(vol6),'orientation':'positive' if vol6>0 else 'negative' if vol6<0 else 'zero','closed_exact_component_count':topo,'bbox_min_m':[float(x) for x in lo],'bbox_max_m':[float(x) for x in hi],'representative_boundary_point_m':[float(x) for x in rep]})
  prepared.append(ci.prepare_point_location(rec));reps.append(rep)
 return out,prepared,reps

src_comps_report,src_prepared,src_reps=comp_summary(source_points,source_faces,source_comps)
# Exact original-row intersections and orientation-boundary details.
source_records=ci._records(source_points,source_faces.tolist())
source_self=ci._audit_pair(source_records,source_records,same_surface=True)
source_topo=ci._validate_closed_oriented_record_topology(source_records)

# Evaluate source component nesting by exact point-location of one representative face point on each other shell.
source_nesting=[]
for i,rep in enumerate(src_reps):
 row={'component':i,'representative_inside_components':[],'location_by_component':[]}
 for j,prepared in enumerate(src_prepared):
  if i==j:continue
  result=ci.point_location_prepared_indexed(rep,prepared)
  row['location_by_component'].append({'component':j,**result})
  if result['location']=='inside':row['representative_inside_components'].append(j)
 source_nesting.append(row)

cp=pickle.loads(CAND.read_bytes());s=cp['surfaces'][133]
cand_points=[tuple(F(x) for x in p) for p in s['vertices_exact_m']]
cand_faces=np.asarray(s['faces'],dtype=np.int64)
cand_global_faces=[tuple(map(int,t)) for t in s['faces_global']]
cand_top=analyze_topology([tuple(float(x) for x in p) for p in s['vertices_float32_m']],cand_faces.tolist())
cand_comps=[list(map(int,c)) for c in cand_top['face_components']]
cand_comps_report,cand_prepared,cand_reps=comp_summary(cand_points,cand_faces,cand_comps)
cand_records=ci._records(cand_points,cand_faces.tolist())
cand_self=ci._audit_pair(cand_records,cand_records,same_surface=True)
cand_topo=ci._validate_closed_oriented_record_topology(cand_records)
cand_nesting=[]
for i,rep in enumerate(cand_reps):
 row={'component':i,'representative_inside_components':[],'location_by_component':[]}
 for j,prepared in enumerate(cand_prepared):
  if i==j:continue
  result=ci.point_location_prepared_indexed(rep,prepared)
  row['location_by_component'].append({'component':j,**result})
  if result['location']=='inside':row['representative_inside_components'].append(j)
 cand_nesting.append(row)

# Source ancestry, role, and exact triangle-equality mapping for each candidate component.
ancestry=[]
parent_ids=list(map(int,s['parent_ids']));parent_faces=list(map(int,s['parent_faces']));roles=list(map(int,s['roles']))
source_comp_by_face={int(fi):ci0 for ci0,fs in enumerate(source_comps) for fi in fs}
for ci0,fs in enumerate(cand_comps):
 parent_distribution={};role_distribution={};mapped_counts={};exact_unchanged=0;bad=[];parentface_union=set()
 for fi in fs:
  sid=parent_ids[fi];sf=parent_faces[fi];role=roles[fi]
  parent_distribution[sid]=parent_distribution.get(sid,0)+1;role_distribution[role]=role_distribution.get(role,0)+1
  if sid==133 and 0<=sf<len(source_faces):
   parentface_union.add(sf);mapped_counts[source_comp_by_face.get(sf,-1)]=mapped_counts.get(source_comp_by_face.get(sf,-1),0)+1
   cand_tri=[cand_points[int(v)] for v in cand_faces[fi]]
   src_tri=[source_points[int(v)] for v in source_faces[sf]]
   if sorted(cand_tri)==sorted(src_tri): exact_unchanged+=1
   else: bad.append({'candidate_face':fi,'source_parent_face':sf,'candidate_global_ids':list(cand_global_faces[fi])})
  else: bad.append({'candidate_face':fi,'source_parent_id':sid,'source_parent_face':sf})
 ancestry.append({'candidate_component':ci0,'candidate_signed_orientation':cand_comps_report[ci0]['orientation'],'face_count':len(fs),'source_parent_stable_id_distribution':parent_distribution,'role_distribution':role_distribution,'raw_source133_component_face_counts':{str(k):v for k,v in mapped_counts.items()},'unique_raw_source_parent_face_count':len(parentface_union),'exact_source_parent_triangle_matches':exact_unchanged,'nonmatching_or_non133_faces':len(bad),'first_nonmatching_examples':bad[:3]})

# Pairwise source component disjointness from the exact row self-intersection audit.
report={'schema':'numi.human.right133-inherited-shell-classification-181.v1','source_payload_path':str(SOURCE),'source_payload_sha256':sha(SOURCE),'source_manifest_path':str(MANIFEST),'source_manifest_sha256':sha(MANIFEST),'binding_loader_path':str(LOADER),'binding_loader_sha256':sha(LOADER),'candidate_path':str(CAND),'candidate_sha256':sha(CAND),'intersection_module_path':str(PUB/'numilab_human/cardiac_cavity_intersections.py'),'intersection_module_sha256':sha(PUB/'numilab_human/cardiac_cavity_intersections.py'),'source_raw_row133':{'vertices':len(raw['positions']),'faces':len(source_faces),'components':src_comps_report,'component_nesting':source_nesting,'whole_row_exact_self_pairs':int(source_self['count']),'whole_row_exact_topology_component_count':source_topo},'candidate_row133':{'vertices':len(s['vertices_exact_m']),'faces':len(cand_faces),'components':cand_comps_report,'component_nesting':cand_nesting,'whole_row_exact_self_pairs':int(cand_self['count']),'whole_row_exact_topology_component_count':cand_topo,'component_source_ancestry':ancestry},'interpretation_boundary':'Exact nested-shell/ancestry evidence classifies whether components are detached. Source metadata does not label an anatomical cavity purpose; geometric nesting alone does not prove the biological intent of each enclosed region.','candidate_admission':False}
out=OUT/'report.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(out),'report_sha256':sha(out),'source_faces':len(source_faces),'source_components':[(x['face_count'],x['orientation'],x['signed_volume_m3']) for x in src_comps_report],'source_nesting':source_nesting,'source_self_pairs':int(source_self['count']),'candidate_components':[(x['face_count'],x['orientation'],x['signed_volume_m3']) for x in cand_comps_report],'candidate_nesting':cand_nesting,'candidate_self_pairs':int(cand_self['count']),'ancestry':ancestry},sort_keys=True),flush=True)
