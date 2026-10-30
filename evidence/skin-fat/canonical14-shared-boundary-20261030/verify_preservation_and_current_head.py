from pathlib import Path
import numpy as np,json,hashlib,csv,sys,importlib.util,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
R=P/'canonical-D-boundary-binding-696/run-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p)as z:return{k:z[k]for k in z.files}
def coord(d):return next(k for k in ('points_world_m_f32','points_world_m','points_world_m_f64')if k in d)
T=time.monotonic();c=json.loads((R/'candidates.json').read_text());checks=[]
packet=P/'protected-149-current-surface-crosswalk-001/run-009-correct-witness-operand-packet/protected-149-targets.csv'
assert sha(packet)=='9dfc74113fc1b4b549d26437a20adefa05766f75c2f38b2853273ebcc0e6a24b'
rows=list(csv.DictReader(packet.open()))
for ca in c:
 assert sha(ca['input_path'])==ca['input_sha256']and sha(ca['path'])==ca['sha256']
 a=load(ca['input_path']);b=load(ca['path'])
 keys=[k for k in a if k in ['points_world_m_f32','points_world_m','points_world_m_f64']]
 expected=np.sort(np.asarray([x['vertex_row']for x in ca['updates']],np.int64))
 for key in keys:
  actual=np.flatnonzero(np.any(a[key]!=b[key],axis=1))
  assert np.array_equal(actual,expected),(ca['name'],key,len(actual),len(expected))
 for k in a:
  if k not in keys:assert np.array_equal(a[k],b[k]),k
 x=b[keys[0]].astype(np.float32);used=set(np.unique(b['faces']).tolist());sites=[];head=0
 for r in rows:
  if not r['stable_attachment_id']:
   if r['role']=='D':
    j=int(r['current_surface_vertex_row']);target=np.asarray([float(r[k])for k in ['target_world_x_m','target_world_y_m','target_world_z_m']],np.float32)
    if ca['name']=='source':assert np.array_equal(x[j],target);head+=1
   continue
  target=np.asarray([float(r[k])for k in ['target_world_x_m','target_world_y_m','target_world_z_m']],np.float32)
  found=[int(j)for j in np.flatnonzero(np.all(x==target,axis=1))if int(j)in used];assert len(found)==1
  sites.append({'id':int(r['stable_attachment_id']),'surface_vertex_row':found[0],'target_float32_exact':True,'incident_face_count':int(np.sum(np.any(b['faces']==found[0],axis=1)))})
 assert len(sites)==14
 checks.append({'candidate':ca['name'],'sha256':ca['sha256'],'changed_vertex_count':len(expected),'face_connectivity_unchanged':True,'original_noncoordinate_arrays_unchanged':True,'outside_coordinates_exact':True,'max_displacement_m':ca['max_displacement_m'],'min_normal_cos':ca['min_face_normal_cos'],'area_ratio_min':ca['area_ratio_min'],'area_ratio_max':ca['area_ratio_max'],'head_D_45_exact':head if ca['name']=='source'else'NOT_RUN_REDUCED_ROW_DOMAIN','canonical14':sites})
(R/'preservation-and-target-check-002.json').write_text(json.dumps({'status':'PASS','checks':checks,'physical_admission':False,'runtime':0,'accepted_steps':0},indent=2)+'\n')
loader=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py'
s=importlib.util.spec_from_file_location('loader696v2',loader);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);pred=m.load_exact_owner()
assert sha(pred.__file__)=='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
source=c[0];d=load(source['path']);dr,_=pred.float32_surface_records(d[coord(d)],d['faces'],source_seam_groups=())
queries=[dr[i]for i in source['changed_face_rows']];reports=[]
for name,path,digest in [
 ('I014',P/'paired-head-reference-fit-684/run-014-witness-local-taper/candidate-I-f32-collapsed-ancestry.npz','7e40a870fffb6c29b34f300951d31e604d874874362ab620fd39f86d05fea36d'),
 ('O392',P/'surface-transfer-checkpoint-392/payload/O/candidate.npz','2958a704db4b8b68422640a5e991c6b51af25f19d525c49adc8a742f0c207d7a')]:
 assert sha(path)==digest
 a=load(path);ar,_=pred.float32_surface_records(a[coord(a)],a['faces'],source_seam_groups=())
 rr=pred._audit_record_pairs(pred._query_surface_aabb(queries,pred._prepare_surface_aabb(ar)),same_surface=False)
 pairs=R/('changed-source-D-vs-'+name+'-pairs.npz');np.savez_compressed(pairs,triangle_pairs=np.asarray(rr['triangle_pairs'],np.int64).reshape(-1,2))
 reports.append({'target':name,'target_path':str(path),'target_sha256':digest,'count':rr['count'],'aabb_candidate_pairs':rr['aabb_candidate_pairs'],'pairs_sha256':sha(pairs),'pairs_file':str(pairs)})
 (R/'head-I-and-O-incremental-exact-gates.json').write_text(json.dumps({'status':'PASS'if all(r['count']==0 for r in reports)else'FAILED','reports':reports,'elapsed_s':time.monotonic()-T,'physical_admission':False,'runtime':0,'accepted_steps':0},indent=2)+'\n')
 print(json.dumps(reports[-1]),flush=True)
 if rr['count']:raise SystemExit(2)

