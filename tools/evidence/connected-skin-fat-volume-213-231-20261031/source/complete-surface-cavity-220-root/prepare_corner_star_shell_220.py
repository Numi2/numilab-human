from pathlib import Path
import argparse,hashlib,importlib.util,json,numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
SOURCE=H/'complete-surface-cavity-217-native/skin-run001/connected-fixed-surface-volume-proposal.npz'
SOURCE_SHA='da043a9e566b4af38eb8a6cb07860da220f771be85b294309f655f530913d46b'
OWNER=H/'transfer_surface_contractions_to_volume_209_root.py'
OWNER_SHA='5d90632e208af37d3a4f4bd6bcaaad3533eec1113e5145cdb3fecfba322dbdbd'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--center',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if sha(SOURCE)!=SOURCE_SHA or sha(OWNER)!=OWNER_SHA:raise SystemExit('pinned source or boundary owner changed')
 if a.out.exists():raise SystemExit('refusing to overwrite '+str(a.out))
 a.out.mkdir(parents=True)
 spec=importlib.util.spec_from_file_location('skin_cavity220_boundary_owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 with np.load(SOURCE,allow_pickle=False) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
 target=463367
 if p.dtype!=np.float32 or int(a.center) not in t[target]:raise SystemExit('source dtype or target-corner check failed')
 rows=np.flatnonzero(np.any(t==a.center,axis=1));local=t[rows]
 if target not in rows or np.unique(mat[rows]).tolist()!=[15]:raise SystemExit('target/material check failed')
 faces=m.boundary(local);gpoints=np.unique(local)
 if not np.array_equal(np.unique(faces),gpoints):raise SystemExit('star contains a nonboundary point')
 lookup=np.full(len(p),-1,np.int64);lookup[gpoints]=np.arange(len(gpoints));local_faces=lookup[faces]
 edges=np.sort(np.concatenate([local_faces[:,[0,1]],local_faces[:,[1,2]],local_faces[:,[2,0]]]),axis=1)
 eu,ec=np.unique(edges,axis=0,return_counts=True)
 directed={}
 for x,y in np.concatenate([local_faces[:,[0,1]],local_faces[:,[1,2]],local_faces[:,[2,0]]]):
  k=(min(int(x),int(y)),max(int(x),int(y)));directed[k]=directed.get(k,0)+(1 if x<y else -1)
 if not(np.all(ec==2) and all(v==0 for v in directed.values())):raise SystemExit('boundary edge chain not closed')
 PAT=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64);allfaces=t[:,PAT].reshape(-1,3);sf=np.sort(allfaces,axis=1);n=len(p);codes=(sf[:,0]*n+sf[:,1])*n+sf[:,2];targetf=np.sort(faces,axis=1);tc=(targetf[:,0]*n+targetf[:,1])*n+targetf[:,2];order=np.argsort(codes);sc=codes[order]
 def orient(x):return 1 if ((int(x[0]>x[1])+int(x[0]>x[2])+int(x[1]>x[2]))%2)==0 else -1
 for j,(tri,code) in enumerate(zip(faces,tc)):
  lo=np.searchsorted(sc,code,'left');hi=np.searchsorted(sc,code,'right');occ=order[lo:hi]
  if hi-lo not in (1,2):raise SystemExit(f'whole-mesh face incidence invalid row {j}')
  si=[k for k in occ if k//4 in rows];oi=[k for k in occ if k//4 not in rows]
  if len(si)!=1 or (hi-lo==2 and len(oi)!=1):raise SystemExit(f'selected/outside incidence mismatch row {j}')
  if orient(allfaces[si[0]])!=orient(tri) or (oi and orient(allfaces[oi[0]])==orient(tri)):raise SystemExit(f'current boundary winding mismatch row {j}')
 lattice=m.m.integer_points(p);vol=0
 for tri in faces:
  x,y,z=[lattice[int(v)] for v in tri];vol+=x[0]*(y[1]*z[2]-y[2]*z[1])-x[1]*(y[0]*z[2]-y[2]*z[0])+x[2]*(y[0]*z[1]-y[1]*z[0])
 if vol<=0:raise SystemExit('exact signed boundary volume is not positive')
 shell=a.out/'skin-corner-star-shell.npz';np.savez_compressed(shell,points_world_m=p[gpoints],faces=local_faces,global_point_rows=gpoints,source_cell_rows=rows,source_boundary_faces_global=faces,target_negative_cell_row=np.asarray(target,np.int64),star_center_global_point_row=np.asarray(a.center,np.int64),material_index=np.asarray(15,np.int32))
 r={'schema':'numi.skin.negative-cavity-220-shell.v1','status':'PASS_CLOSED_CURRENT_BOUNDARY_STAR','physical_admission':False,'source_candidate_path':str(SOURCE),'source_candidate_sha256':SOURCE_SHA,'source_owner_path':str(OWNER),'source_owner_sha256':OWNER_SHA,'target_negative_cell_row':target,'target_negative_corners_global_point_rows':t[target].tolist(),'star_center_global_point_row':int(a.center),'selected_source_cell_rows':len(rows),'selected_source_point_rows':len(gpoints),'boundary_face_rows':len(faces),'material_indices':[15],'directed_boundary_edge_closed':True,'current_mesh_boundary_and_outside_opposition':True,'exact_signed_six_volume_positive':True,'exact_signed_six_volume_lattice_digits':len(str(abs(vol))),'shell_path':str(shell),'shell_sha256':sha(shell),'shell_point_rows':len(gpoints),'shell_face_rows':len(faces)}
 (a.out/'preflight-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
