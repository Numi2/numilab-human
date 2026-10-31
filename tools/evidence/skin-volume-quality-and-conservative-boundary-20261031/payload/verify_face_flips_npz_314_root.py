from pathlib import Path
import json,hashlib,importlib.util,time,numpy as np
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cert198',H/'fixed_boundary_volume_flip_198_root.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def boundary(t):
 f=t[:,m.PAT].reshape(-1,3);sf=np.sort(f,axis=1);order=np.lexsort((sf[:,2],sf[:,1],sf[:,0]));ss=sf[order]
 st=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];ct=np.diff(np.r_[st,len(f)])
 signs=1-2*(((f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8))%2)
 sm=np.add.reduceat(signs[order],st)
 assert np.all(ct<=2) and np.all(sm[ct==2]==0)
 return np.c_[ss[st[ct==1]],sm[ct==1]]
def run():
 tic=time.monotonic();inp=H/'interior-quality-conditioning-308-banks/skin-run002/conditioned-volume-proposal.npz';out=H/'face-flips-npz-313-root/skin-run001/fixed-boundary-volume-proposal.npz'
 assert sha(inp)=='309d41b86dc0380f4ee64ff9fec6c2caec64b8fe5bd26d434cfde36f7d7b759c'
 assert sha(out)=='5d89d2700ddd87f734e638105249f8b88365283b17c23cb41bf18713f22eb061'
 with np.load(inp) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
 with np.load(out) as z:v={k:z[k] for k in z.files}
 assert np.array_equal(p.view('u4'),v['points_world_m'].view('u4'))
 parents=v['replacement_parent_cell_rows'];new=v['replacement_tetrahedra'];repm=[];seen=set()
 for rows,nt in zip(parents,new):
  verts=set(map(int,t[rows].ravel()));assert not verts & seen;seen.update(verts)
  assert len(np.unique(mat[rows]))==1
  assert m.certify(p,t[rows],nt)==(True,'PASS')
  assert np.all(m.passes(*m.measurements(p,nt)))
  assert np.any(~m.passes(*m.measurements(p,t[rows])))
  repm.extend([mat[rows[0]]]*len(nt))
 keep=np.ones(len(t),bool);keep[parents.ravel()]=False
 assert np.array_equal(np.flatnonzero(keep),v['retained_input_cell_rows'])
 assert np.array_equal(np.vstack([t[keep],new.reshape(-1,4)]),v['tetrahedra'])
 assert np.array_equal(np.r_[mat[keep],repm],v['material_indices'])
 assert np.array_equal(boundary(t),boundary(v['tetrahedra']))
 d,q=m.measurements(p,v['tetrahedra']);assert np.all(d>0)
 r={'status':'PASS_INDEPENDENT_SOURCE198_FACE_FLIP_REPLAY','source_sha256':sha(Path(__file__)),'input':str(inp),'input_sha256':sha(inp),'candidate':str(out),'candidate_sha256':sha(out),'operations':len(parents),'complete_boundary_faces':len(boundary(t)),'complete_oriented_boundary_identical':True,'coordinates_bit_identical':True,'material_preserved':True,'all_exact_local_volume_and_orientation_checks':True,'nonpositive_cells':0,'native_quality_failures':int(np.count_nonzero(~m.passes(d,q))),'physical_admission':False,'native_steps':0,'elapsed_s':time.monotonic()-tic}
 dest=H/'face-flips-npz-313-root/independent-314-001';dest.mkdir(exist_ok=False);(dest/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':run()
