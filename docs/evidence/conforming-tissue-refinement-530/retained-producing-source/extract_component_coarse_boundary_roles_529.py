from pathlib import Path
import numpy as np,json,hashlib,argparse
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def extract(base,component,view,fat):
 results={}
 for role in ['O','I','D']:
  domain=np.asarray(base['coarse_boundary_faces_'+role],dtype=np.int64)
  keyed={tuple(sorted(map(int,t))):i for i,t in enumerate(domain)}
  if len(keyed)!=len(domain):raise ValueError('duplicate coarse boundary face')
  fmap={i:keyed[k] for i,t in enumerate(fat['faces']) if (k:=tuple(sorted(map(int,t)))) in keyed}
  rows=[(i,int(p),fmap[int(p)]) for i,(o,p) in enumerate(zip(view['face_parent_source_original_id'],view['face_parent_source_face_id'])) if int(o)==62000121 and int(p) in fmap]
  rr=np.asarray(rows,dtype=np.int64).reshape(-1,3)
  cp=component['points_world_m_f64'];cf=component['faces_f64'];tri=cp[cf[rr[:,0]]] if len(rr) else np.zeros((0,3,3))
  areas=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
  results[role]={'rows':rr,'area_m2':float(areas.sum()),'unique_coarse_face_rows':np.unique(rr[:,2]).tolist()}
 return results
def main():
 ap=argparse.ArgumentParser()
 for name in ['base','component','component-view','fat-view','control-I','fine-D','out']:ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 def load(p):
  with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 b,c,v,f=map(load,[a.base,a.component,a.component_view,a.fat_view]);r=extract(b,c,v,f);control=load(a.control_I)
 assert np.array_equal(np.unique(control['source002_I_boundary_face_rows']),r['I']['unique_coarse_face_rows'])
 assert len(r['I']['rows'])==548 and len(r['I']['unique_coarse_face_rows'])==28
 assert len(r['D']['rows'])==313 and len(r['D']['unique_coarse_face_rows'])==31
 fine=load(a.fine_D)
 data={role+'_'+k:np.asarray(val) for role,rr in r.items() for k,val in rr.items()}
 np.savez_compressed(a.out/'component-coarse-role-seeds.npz',**data)
 receipt={'status':'exact_component_to_original_boundary_role_map','input_sha256':{name:sha(getattr(a,name)) for name in ['base','component','component_view','fat_view','control_I','fine_D']},'roles':{role:{k:val for k,val in rr.items() if k!='rows'}|{'component_subfaces':len(rr['rows'])} for role,rr in r.items()},'I_regression_matches_028_exactly':True,'method':'component face parent original ID62000121 and exact sorted original node triple, no proximity or donor-only matching','fine_D_path':str(a.fine_D),'fine_D_faces_sha256_int64_little_endian_C':hashlib.sha256(np.asarray(fine['faces'],dtype='<i8').tobytes()).hexdigest(),'fine_D_points_sha256_float32_little_endian_C':hashlib.sha256(np.asarray(fine['points_world_m'],dtype='<f4').tobytes()).hexdigest(),'candidate_or_binding_changes':False,'physical_admission':False}
 (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
