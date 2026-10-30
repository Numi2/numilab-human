"""Exact current D refinement audit; inherits only bitwise retained geometry."""
from pathlib import Path
import argparse,json,hashlib,numpy as np,sys,importlib.util,time
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');ROOT=H.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def imp(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def cyclic(f):
 f=np.asarray(f,np.int64);a=np.argmin(f,axis=1);return np.take_along_axis(f,(a[:,None]+np.arange(3))%3,axis=1)
def keys(f):return np.ascontiguousarray(cyclic(f)).view(np.dtype([('a','<i8'),('b','<i8'),('c','<i8')])).reshape(-1)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--D',type=Path,required=True);ap.add_argument('--D-sha256',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists();assert sha(a.D)==a.D_sha256;a.out.mkdir();started=time.monotonic()
 inherited=H/'inferred-D-boundary-exact-delta-128-root/report.json';ip=json.loads(inherited.read_text());assert ip['status']=='PASS_EXACT_CURRENT_D_ONE_ULP_DELTA'
 oldpath=Path(ip['sources']['D']['path']);assert sha(oldpath)==ip['sources']['D']['sha256'];oldD=load(oldpath);D=load(a.D)
 assert D['points_world_m'].dtype==np.float32 and np.isfinite(D['points_world_m']).all()
 assert len(D['points_world_m'])==len(oldD['points_world_m'])+8
 assert np.array_equal(D['points_world_m'][:len(oldD['points_world_m'])].view(np.uint32),oldD['points_world_m'].view(np.uint32))
 assert D['points_world_m'].tobytes()==D['points_world_m_f32'].tobytes()
 oldkeys=keys(oldD['faces']);newkeys=keys(D['faces']);order=np.argsort(oldkeys);idx=np.searchsorted(oldkeys[order],newkeys);matched=(idx<len(oldkeys));matched[matched]&=oldkeys[order][idx[matched]]==newkeys[matched];fresh=np.flatnonzero(~matched);oldused=order[idx[matched]]
 missing=np.setdiff1d(np.arange(len(oldkeys)),oldused);assert len(fresh)==32 and len(missing)==16 and len(np.unique(oldused))==len(oldused)
 assert np.array_equal(np.unique(D['faces']),np.arange(len(D['points_world_m'])))
 v=imp('verify034refine137',H/'quality-proposal-verification-034/verify_quality_proposal_034.py')
 assert v.directed_boundary(oldD['faces'])==v.directed_boundary(D['faces'])
 q=D['points_world_m'][D['faces'][fresh]].astype(float);areas=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1);assert np.all(areas>0)
 assert sha(v.OWNER)==v.OWNER_SHA and sha(v.BONE)==v.BONE_SHA
 sys.path.insert(0,str(v.OWNER.parent.parent));import numilab_human.cardiac_cavity_intersections as pred
 bone=imp('bone137',v.BONE);paths={r:Path(ip['sources'][r]['path'])for r in ['O','I']};paths['D']=a.D
 for r in ['O','I']:assert sha(paths[r])==ip['sources'][r]['sha256']
 report={'status':'STARTED','physical_admission':False,'runtime':0,'accepted_steps':0,'inherited_complete_geometry_proof':{'path':str(inherited),'sha256':sha(inherited)},'retained_D_vertices_bitwise_unchanged':True,'appended_D_vertices':8,'retained_D_faces':int(matched.sum()),'replaced_D_source_face_rows':missing.tolist(),'new_D_face_rows':fresh.tolist(),'unchanged_source_oriented_boundary_edges':True,'new_D_min_double_area_m2':float(areas.min()),'sources':{r:{'path':str(p),'sha256':sha(p)}for r,p in paths.items()},'all149_attachment_coordinates_unchanged':[]}
 surfaces={r:(D if r=='D'else load(paths[r]))for r in ['O','I','D']}
 for role,t_id,row in ip['all149_attachment_coordinates_unchanged']:
  aa=surfaces[role];ref=oldD if role=='D'else aa
  assert aa['points_world_m'][row].tobytes()==ref['points_world_m'][row].tobytes();report['all149_attachment_coordinates_unchanged'].append([role,t_id,row])
 assert len(report['all149_attachment_coordinates_unchanged'])==149
 allrecords=[];selected=set();facebase=pointbase=0
 for r in ['O','I','D']:
  aa=surfaces[r];rr,_=pred.float32_surface_records(aa['points_world_m'].astype(float),aa['faces'],source_seam_groups=())
  allrecords.extend((x[0],x[1],x[2],facebase+i,tuple(pointbase+int(y)for y in x[4]))for i,x in enumerate(rr))
  if r=='D':selected.update(map(int,fresh+facebase))
  facebase+=len(rr);pointbase+=len(aa['points_world_m'])
 assert len(allrecords)==646175;report['complete_current_surface_face_count']=len(allrecords)
 report['changed_vs_complete_current_surfaces']=pred._audit_self_pairs_touching_faces(allrecords,selected);assert report['changed_vs_complete_current_surfaces']['count']==0
 report['status']='CHECKING_ALL185_BOTH_DOMAINS';(a.out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
 bones,breceipt=bone.load_bone_barrier_records(pred,root=ROOT,vertex_offset=pointbase,expected_native_sha256='e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5',expected_manifest_sha256='0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c')
 assert len(bones)==938272
 report['changed_vs_all185_both_domains']=pred._audit_pair_prepared_first(pred._prepare_surface_aabb(bones),[r for r in allrecords if int(r[3])in selected],same_surface=False);assert report['changed_vs_all185_both_domains']['count']==0
 report['bone_sources']=breceipt;report['status']='PASS_EXACT_CURRENT_D_BOUNDARY_REFINEMENT_DELTA';report['elapsed_s']=time.monotonic()-started
 report['interpretation']='Complete coverage inherits only bitwise retained oriented face geometry from the pinned complete-coverage chain and exactly checks every new face against all current surfaces and all185 bones in both domains. No mass or native motion claim.'
 (a.out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'new_faces':len(fresh),'elapsed_s':report['elapsed_s'],'surface_crossings':report['changed_vs_complete_current_surfaces']['count'],'bone_crossings':report['changed_vs_all185_both_domains']['count']}),flush=True)
if __name__=='__main__':main()
