"""Check retained endpoint determinant semantics against initialized and rest frames.
Postprocessing only. Does not change physics, gates, or promote a run.
"""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json
import numpy as np

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def audit(path,owner_path):
 spec=importlib.util.spec_from_file_location('retained_endpoint_owner',owner_path)
 owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
 z=owner.capture(path);n=len(z['rest']);nt=len(z['tets'])
 r0=np.empty((n,3));u0=np.empty((n,3));raw_j=np.empty(nt)
 seen_n=np.zeros(n,bool);seen_t=np.zeros(nt,bool)
 with gzip.open(path,'rt') as f:
  for line in f:
   if 'resting_FEM_node,' in line:
    row=owner.kv(line.split('resting_FEM_node,',1)[1],',');i=int(row['global'])
    assert not seen_n[i];seen_n[i]=True
    r0[i]=[owner.f32(row['initial_rest_'+a]) for a in 'xyz']
    mode=int(float(row['initial_reference_mode']));assert mode in (0,1)
    u0[i]=([owner.f32(row['initial_reference_u_hi_'+a])+owner.f32(row['initial_reference_u_lo_'+a]) for a in 'xyz'] if mode else [owner.f32(row['initial_'+a])-r0[i,j] for j,a in enumerate('xyz')])
   elif 'resting_FEM_tet,' in line:
    row=owner.kv(line.split('resting_FEM_tet,',1)[1],',');i=int(row['global'])
    assert not seen_t[i];seen_t[i]=True;raw_j[i]=float(row['J'])
 assert seen_n.all() and seen_t.all()
 names={v:k for k,v in z['objects'].items()};regions={};witnesses=[]
 for start in range(0,nt,8192):
  stop=min(start+8192,nt);t=z['tets'][start:stop]
  def edges(x):return np.swapaxes(x[t][:,1:]-x[t][:,:1],1,2)
  dm0=edges(r0);dm1=edges(z['rest']);du0=edges(u0);du1=edges(z['u']);dx0=dm0+du0;dx1=dm1+du1
  det0=np.linalg.det(dx0);det1=np.linalg.det(dx1)
  assert np.all(np.isfinite(det0)) and np.all(det0!=0)
  relative_j=det1/det0;absolute_j=det1/np.linalg.det(dm1)
  initial_f=dx0@np.linalg.inv(dm0);current_f=dx1@np.linalg.inv(dm1)
  incremental_f=dx1@np.linalg.inv(dx0)
  def green(f):return np.linalg.norm(.5*(np.swapaxes(f,1,2)@f-np.eye(3)),axis=(1,2))
  initial_e=green(initial_f);current_e=green(current_f);incremental_e=green(incremental_f)
  labels=z['treg'][start:stop];obs=z['tobj'][start:stop];groups=np.where(labels>=0,labels,1000000+obs.astype(np.int32))
  error=np.abs(relative_j-raw_j[start:stop]);wrong=np.abs(absolute_j-raw_j[start:stop])
  for label in np.unique(groups):
   m=groups==label;key='region:'+str(label) if label<1000000 else 'object:'+names[int(label-1000000)]
   row=regions.setdefault(key,dict(cells=0,maximum_native_J_replay_error=0.,maximum_rest_J_difference_from_emitted=0.,initial_green_max=0.,current_rest_green_max=0.,incremental_green_max=0.))
   row['cells']+=int(m.sum())
   for k,v in [('maximum_native_J_replay_error',error),('maximum_rest_J_difference_from_emitted',wrong),('initial_green_max',initial_e),('current_rest_green_max',current_e),('incremental_green_max',incremental_e)]:row[k]=max(row[k],float(v[m].max()))
   selected=np.flatnonzero(m)[np.argmax(current_e[m])]
   if 'peak_strain_witness' not in row or float(current_e[selected])>row['peak_strain_witness']['green_norm']:
    dm=dm1[selected];sv=np.linalg.svd(dm,compute_uv=False);g=start+int(selected)
    row['peak_strain_witness']=dict(tet=g,nodes=z['tets'][g].tolist(),green_norm=float(current_e[selected]),initial_green_norm=float(initial_e[selected]),incremental_green_norm=float(incremental_e[selected]),rest_edge_singular_values_m=sv.tolist(),rest_edge_condition=float(sv[0]/sv[-1]),current_displacement_edge_norm_m=float(np.linalg.norm(du1[selected])),incremental_displacement_edge_norm_m=float(np.linalg.norm(du1[selected]-du0[selected])),emitted_initialized_J=float(raw_j[g]))

  local=int(np.argmax(wrong));i=start+local
  witnesses.append(dict(tet=i,region=int(z['treg'][i]),emitted_J=float(raw_j[i]),replayed_initialized_J=float(relative_j[local]),rest_frame_J=float(absolute_j[local]),initial_green=float(initial_e[local]),current_rest_green=float(current_e[local]),incremental_green=float(incremental_e[local])))
 witnesses=sorted(witnesses,key=lambda x:abs(x['rest_frame_J']-x['emitted_J']),reverse=True)[:8]
 return dict(input=dict(path=str(path),sha256=sha(path)),capture_owner=dict(path=str(owner_path),sha256=sha(owner_path)),nodes=n,tetrahedra=nt,maximum_initial_reference_displacement_m=float(np.linalg.norm(u0,axis=1).max()),maximum_rest_coordinate_change_m=float(np.linalg.norm(z['rest']-r0,axis=1).max()),regions=regions,witnesses=witnesses,physical_admission=False,scope='Determinant/strain reference semantics only. Neither initialized-incremental strain nor rest-frame strain replaces material energy/tangent checks or existing force gates.')

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('capture',type=Path);ap.add_argument('--capture-owner',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
 r=audit(a.capture,a.capture_owner);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(nodes=r['nodes'],tetrahedra=r['tetrahedra'],maximum_initial_reference_displacement_m=r['maximum_initial_reference_displacement_m'],maximum_rest_coordinate_change_m=r['maximum_rest_coordinate_change_m'],witnesses=r['witnesses']),indent=2))
