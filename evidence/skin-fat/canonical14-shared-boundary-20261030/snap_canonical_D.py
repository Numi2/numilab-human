from pathlib import Path
import numpy as np,json,hashlib,time,importlib.util,sys,signal,csv
from scipy.spatial import cKDTree
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
OUT=Path(__file__).resolve().parent/'run-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def progress(stage,**kw):
 d={'stage':stage,'elapsed_s':time.monotonic()-T,**kw};(OUT/'progress.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
def arrays(p):
 with np.load(p)as z:return{k:z[k] for k in z.files}
def coords(d):
 return next(k for k in ['points_world_m_f32','world_points_f32','points','world_points','points_world_m','points_world_m_f64']if k in d)
def main():
 global T
 T=time.monotonic();OUT.mkdir(exist_ok=False)
 inputs=json.loads((OUT.parent/'nearest-used-vertices.json').read_text())
 packet=P/'protected-149-current-surface-crosswalk-001/run-009-correct-witness-operand-packet/protected-149-targets.csv'
 if sha(packet)!='9dfc74113fc1b4b549d26437a20adefa05766f75c2f38b2853273ebcc0e6a24b':raise ValueError('canonical packet changed')
 loader=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py'
 s=importlib.util.spec_from_file_location('loader696',loader);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);pred=m.load_exact_owner()
 if sha(pred.__file__)!='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d':raise ValueError('exact owner changed')
 # Bind every update to the immutable target packet, not to the diagnostic nearest JSON.
 canonical={int(r['stable_attachment_id']):np.asarray([float(r[k])for k in ['target_world_x_m','target_world_y_m','target_world_z_m']],np.float32).astype(float)for r in csv.DictReader(packet.open())if r['stable_attachment_id']}
 candidates=[]
 for inp in inputs:
  path=Path(inp['path'])
  if sha(path)!=inp['sha256']:raise ValueError('D input changed')
  a=arrays(path);key=coords(a);x=np.asarray(a[key],np.float32).astype(float);f=np.asarray(a['faces'],np.int64);y=x.copy();snaps=[]
  tree=cKDTree(x[np.unique(f)]);used=np.unique(f)
  for row in inp['rows']:
   target=canonical[row['id']]
   if not np.array_equal(target,np.asarray(row['target_m'])):raise ValueError('target packet mismatch')
   dist,near=tree.query(target);j=int(used[near])
   if j!=row['nearest_vertex_row']or dist>2e-7:raise ValueError('source vertex correspondence or 0.2um repair bound failed')
   if dist:
    y[j]=target;snaps.append({'id':row['id'],'vertex_row':j,'before_m':x[j].tolist(),'after_m':target.tolist(),'displacement_m':float(dist)})
  changed=np.flatnonzero(np.isin(f,[s['vertex_row']for s in snaps]).any(axis=1));u=f[changed]
  old=np.cross(x[u[:,1]]-x[u[:,0]],x[u[:,2]]-x[u[:,0]])
  new=np.cross(y[u[:,1]]-y[u[:,0]],y[u[:,2]]-y[u[:,0]])
  co=np.einsum('ij,ij->i',old,new)/(np.linalg.norm(old,axis=1)*np.linalg.norm(new,axis=1))
  if np.any(~np.isfinite(co))or np.min(co)<.999 or np.any(np.linalg.norm(new,axis=1)==0):raise ValueError('local shape/orientation failed')
  for k,v in list(a.items()):
   if k in ['points_world_m_f32','world_points_f32','points','world_points','points_world_m','points_world_m_f64']and v.shape==x.shape:a[k]=y.astype(v.dtype)
  a['canonical_attachment_shared_vertex_snap_json']=np.asarray([json.dumps({'input_path':str(path),'input_sha256':inp['sha256'],'canonical_packet_sha256':sha(packet),'max_displacement_bound_m':2e-7,'updates':snaps},separators=(',',':'))])
  dest=OUT/(inp['name']+'-D-canonical-attached.npz');np.savez_compressed(dest,**a)
  candidates.append({'name':inp['name'],'path':str(dest),'sha256':sha(dest),'input_path':str(path),'input_sha256':inp['sha256'],'changed_face_rows':changed.tolist(),'snap_count':len(snaps),'max_displacement_m':max(s['displacement_m']for s in snaps),'min_face_normal_cos':float(co.min()),'area_ratio_min':float((np.linalg.norm(new,axis=1)/np.linalg.norm(old,axis=1)).min()),'area_ratio_max':float((np.linalg.norm(new,axis=1)/np.linalg.norm(old,axis=1)).max()),'updates':snaps})
  progress('candidate emitted',name=inp['name'],changed_faces=len(changed),sha256=sha(dest))
 (OUT/'candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
 source=candidates[0];a=arrays(source['path']);x=np.asarray(a[coords(a)],np.float32).astype(float);f=a['faces'];changed=set(source['changed_face_rows'])
 dr,_=pred.float32_surface_records(x,f,source_seam_groups=())
 queries=[dr[i]for i in sorted(changed)];index=pred._prepare_surface_aabb(dr)
 reports=[]
 def audit(name,pairs,same):
  progress('exact audit',name=name)
  r=pred._audit_record_pairs(pairs,same_surface=same);dest=OUT/(name+'-pairs.npz');np.savez_compressed(dest,triangle_pairs=np.asarray(r['triangle_pairs'],np.int64).reshape(-1,2))
  summary={k:v for k,v in r.items()if k!='triangle_pairs'};summary.update(name=name,pairs_path=str(dest),pairs_sha256=sha(dest));reports.append(summary)
  (OUT/'exact-gates.json').write_text(json.dumps(reports,indent=2)+'\n')
  progress('exact audit complete',name=name,count=r['count'],aabb_candidates=r['aabb_candidate_pairs'])
  if r['count']:raise ValueError(name+' introduced '+str(r['count'])+' pairs')
 audit('changed-source-D-self',((a,b)for a,b in pred._query_surface_aabb(queries,index)if a[3]!=b[3]and(b[3]not in changed or b[3]>a[3])),True)
 ip=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz'
 iz=arrays(ip);ir,_=pred.float32_surface_records(np.asarray(iz[coords(iz)],np.float32).astype(float),iz['faces'],source_seam_groups=())
 audit('changed-source-D-vs-I',pred._query_surface_aabb(queries,pred._prepare_surface_aabb(ir)),False)
 bp=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz';b=arrays(bp)
 br,_=pred.float32_surface_records(b['points_world_f32'],b['faces'],source_seam_groups=())
 audit('changed-source-D-vs-all185',pred._query_surface_aabb(queries,pred._prepare_surface_aabb(br)),False)
 # Keep unchanged-domain certification explicit; reduced baseline still requires its full audit.
 final={'status':'PASS_SOURCE_D_INCREMENTAL_GATES','candidates':candidates,'exact_gates':reports,'baseline_D_certificate':str(P/'whole-D-native-bone-interface-628/run-001/report.json'),'baseline_DI_certificate':str(P/'whole-D-fine-I-interface-627/run-001/report.json'),'baseline_D_self_is_retained_evidence':True,'reduced_D_full_geometry_gates':'NOT_RUN','native_volume_binding':'NOT_RUN','physical_admission':False,'runtime':0,'accepted_steps':0,'elapsed_s':time.monotonic()-T}
 (OUT/'report.json').write_text(json.dumps(final,indent=2)+'\n');progress('done',status=final['status'])
if __name__=='__main__':
 try:main()
 except Exception as e:
  OUT.mkdir(exist_ok=True);(OUT/'failure.json').write_text(json.dumps({'error':repr(e),'traceback':__import__('traceback').format_exc(),'runtime':0,'accepted_steps':0},indent=2)+'\n');raise
