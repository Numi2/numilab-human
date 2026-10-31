from pathlib import Path
import argparse,hashlib,json,numpy as np,importlib.util
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
INPUT=H/'complete-surface-cavity-217-native/skin-run001/connected-fixed-surface-volume-proposal.npz'
INPUT_SHA='da043a9e566b4af38eb8a6cb07860da220f771be85b294309f655f530913d46b'
QUALITY=H/'fixed_boundary_volume_flip_198_root.py'
QUALITY_SHA='650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0'
PAT=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shell',type=Path,required=True);ap.add_argument('--milp-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if sha(INPUT)!=INPUT_SHA or sha(QUALITY)!=QUALITY_SHA:raise SystemExit('pinned input or native quality owner changed')
 if a.out.exists():raise SystemExit('refusing to overwrite '+str(a.out))
 shellp=a.shell;mp=a.milp_dir/'cavity-positive-proposal.npz';rp=a.milp_dir/'report.json'
 with np.load(INPUT,allow_pickle=False) as z:basep=z['points_world_m'];baset=z['tetrahedra'];basem=z['material_indices']
 with np.load(shellp,allow_pickle=False) as z:sh={k:z[k] for k in z.files}
 with np.load(mp,allow_pickle=False) as z:prop={k:z[k] for k in z.files}
 report=json.loads(rp.read_text())
 if report.get('status')!='PASS_EXACT_ORIENTED_CAVITY_TETRAHEDRALIZATION' or report.get('physical_admission') is not False:raise SystemExit('MILP receipt is not the bounded exact PASS')
 if report.get('source_sha256')!=sha(shellp) or report.get('proposal_sha256')!=sha(mp):raise SystemExit('MILP receipt hash mismatch')
 if sh['material_index'].item()!=15 or not np.all(prop['material_indices']==15):raise SystemExit('Skin material mismatch')
 g=sh['global_point_rows'].astype(np.int64);localp=sh['points_world_m'];tloc=prop['tetrahedra'].astype(np.int64)
 if not np.array_equal(localp,basep[g]):raise SystemExit('local shell points do not exactly reconstruct source point rows')
 if np.any(tloc<0) or np.any(tloc>=len(g)):raise SystemExit('proposal references unknown local points')
 tg=g[tloc]
 if len(np.unique(np.sort(tg,axis=1),axis=0))!=len(tg):raise SystemExit('duplicate global patch tetrahedron')
 newfaces=tg[:,PAT].reshape(-1,3);fs=np.sort(newfaces,axis=1);fc=(fs[:,0]*len(basep)+fs[:,1])*len(basep)+fs[:,2];order=np.argsort(fc);sc=fc[order];boundary=[]
 for k in np.unique(fc):
  ids=order[np.searchsorted(sc,k,'left'):np.searchsorted(sc,k,'right')]
  if len(ids)==1:boundary.append(newfaces[ids[0]])
 if len(boundary)!=len(sh['source_boundary_faces_global']):raise SystemExit('patch boundary face count changed')
 def canon_tri(x):return min(tuple(x),tuple(np.roll(x,1)),tuple(np.roll(x,2)))
 if sorted(map(canon_tri,boundary))!=sorted(map(canon_tri,sh['source_boundary_faces_global'])):raise SystemExit('global patch boundary differs from preserved current boundary')
 rows=sh['source_cell_rows'].astype(np.int64)
 if not np.all(basem[rows]==15) or int(sh['target_negative_cell_row']) not in rows:raise SystemExit('retired row/material mismatch')
 sp=importlib.util.spec_from_file_location('quality198_patch220',QUALITY);q=importlib.util.module_from_spec(sp);sp.loader.exec_module(q)
 det,nd=q.measurements(localp,tloc);passed=q.passes(det,nd)
 if not np.all(passed):raise SystemExit('selected cavity tets fail native quality owner')
 if len(det)!=len(tloc):raise SystemExit('quality row count mismatch')
 a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,global_point_rows=g,global_tetrahedra=tg,retired_source_cell_rows=rows,material_indices=np.full(len(tg),15,np.int32),points_world_m=localp)
 receipt={'schema':'numi.skin.cavity-220.global-patch.v1','status':'PASS_LOCAL_PATCH_RECONSTRUCTION','physical_admission':False,'input_candidate_path':str(INPUT),'input_candidate_sha256':INPUT_SHA,'local_shell_path':str(shellp),'local_shell_sha256':sha(shellp),'milp_report_path':str(rp),'milp_report_sha256':sha(rp),'local_proposal_path':str(mp),'local_proposal_sha256':sha(mp),'native_quality_owner_path':str(QUALITY),'native_quality_owner_sha256':QUALITY_SHA,'target_negative_cell_row':int(sh['target_negative_cell_row']),'star_center_global_point_row':int(sh['star_center_global_point_row']),'retired_source_cell_rows':len(rows),'new_tetrahedra':len(tg),'material_index':15,'global_point_row_map_exact':True,'current_boundary_preserved':True,'new_tets_pass_native_quality':True,'new_tet_min_normalized_determinant':float(nd.min()),'new_tet_min_volume_m3':float((det/6).min()),'output_patch_path':str(a.out),'output_patch_sha256':sha(a.out)}
 outreceipt=a.out.with_suffix('.receipt.json');outreceipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
