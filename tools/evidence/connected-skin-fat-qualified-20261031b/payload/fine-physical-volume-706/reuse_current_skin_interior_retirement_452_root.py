from pathlib import Path
import numpy as np,json,hashlib,importlib.util,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=mod('boundary452',H/'transfer_surface_contractions_indexed_379_root.py');m=T.m
start=time.monotonic();out=H/'current-skin-interior-retirement-452-root';out.mkdir(exist_ok=False)
ip=H/'outer-O-witness-driven-sequence-448-root/iteration002/qualification435001/skin-target437/skin-bound-current-O-target-volume-proposal.npz'
assert sha(ip)=='7822cd0cd55c346d0c3005cc4f3675bc10f85d51c137893bb1b28aec594da54f'
command=[PY,str(H/'retire_inferred_interior_node_stars_324_root.py'),'--input',str(ip),'--input-sha256',sha(ip),'--source-node',str(H/'combined-child-handoff-142/skin.node'),'--registration-map',str(H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'),'--output',str(out/'run001'),'--seconds','35','--maximum-operations','10000']
with (out/'owner.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
rp=out/'run001/report.json';r=json.loads(rp.read_text());npz=Path(r['output']);assert sha(npz)==r['output_sha256']
with np.load(ip) as z:base={k:z[k] for k in z.files}
with np.load(npz) as z:new={k:z[k] for k in z.files}
p=base['points_world_m'];t=base['tetrahedra'];mat=base['material_indices'];assert np.array_equal(p.view(np.uint32),new['points_world_m'].view(np.uint32))
ret=new['retained_input_cell_rows'];oldr=new['replacement_parent_cell_rows'];oo=new['replacement_parent_offsets'];nt=new['replacement_tetrahedra'];no=new['replacement_tet_offsets'];keep=np.ones(len(t),bool);keep[oldr]=False
assert np.array_equal(ret,np.flatnonzero(keep));assert np.array_equal(new['tetrahedra'],np.vstack([t[keep],nt]))
nm=[];occupied=set()
for j,op in enumerate(r['operations']):
 rows=oldr[oo[j]:oo[j+1]];child=nt[no[j]:no[j+1]];assert np.array_equal(rows,op['old_rows'])
 assert not(set(map(int,rows))&occupied);occupied.update(map(int,rows));assert len(np.unique(mat[rows]))==1
 assert m.certify(p,t[rows],child)==(True,'PASS')
 assert np.count_nonzero(~m.passes(*m.measurements(p,child)))<np.count_nonzero(~m.passes(*m.measurements(p,t[rows])))
 nm.extend([mat[rows[0]]]*len(child))
assert np.array_equal(new['material_indices'],np.r_[mat[keep],np.asarray(nm,dtype=mat.dtype)])
assert np.array_equal(T.boundary_identity(T.boundary(t)),T.boundary_identity(T.boundary(new['tetrahedra'])))
assert np.array_equal(T.boundary_identity(new['desired_boundary_faces']),T.boundary_identity(T.boundary(new['tetrahedra'])))
det,quality=m.measurements(p,new['tetrahedra']);assert np.all(det>0)
proof={'status':'PASS_CURRENT_SKIN_INTERIOR_RETIREMENT_INDEPENDENT_REPLAY','physical_admission':False,'native_steps':0,'input':str(ip),'input_sha256':sha(ip),'candidate':str(npz),'candidate_sha256':sha(npz),'owner_report':str(rp),'owner_report_sha256':sha(rp),'bad_before':r['bad_cells_before'],'bad_after':r['bad_cells_after'],'operations':r['passing_operations'],'coordinates_bit_identical':True,'complete_oriented_boundary_preserved':True,'source330_mass_unchanged':True,'new_cell_mass_mapping':'REQUIRED','operator_sha256':sha(H/'retire_inferred_interior_node_stars_324_root.py'),'producer_sha256':sha(__file__),'command':command,'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
