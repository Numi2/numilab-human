
from pathlib import Path
import ast,py_compile,hashlib,json
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
regsrc=(H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py').read_text()
tree=ast.parse(regsrc);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='registered_source_rows_from_map')
fn=ast.get_source_segment(regsrc,node)
fn=fn.replace('registered_source_rows_from_map','registered_fat_source_rows').replace('"skin"','"fat"').replace('"O"','"D"').replace('Skin','Fat').replace('skin TetGen','fat TetGen')
assert 'if x["source_partition"] != "fat"' in fn and '("D", "I")' in fn
s=(H/'condition_fixed_skin_interior_LP_455_root.py').read_text()
s=s.replace('import argparse,importlib.util,json,hashlib,time,collections','import argparse,importlib.util,json,hashlib,time,collections,csv')
s=s.replace('def run(a):',fn+'\ndef run(a):')
s=s.replace("H/'combined-child-handoff-142/skin.node'","H/'combined-child-handoff-142/fat.node'")
s=s.replace('R.registered_source_rows_from_map(registration,sp)','registered_fat_source_rows(registration,sp)')
a="boundary=T.boundary(t);fixed=np.zeros(len(p),bool);fixed[np.unique(boundary)]=True;fixed[:len(sp)]|=reg"
b="""boundary=T.boundary(t);fixed=np.zeros(len(p),bool);fixed[np.unique(boundary)]=True;fixed[:len(sp)]|=reg
 protection_records=[]
 if a.protected_cell_map:
  assert sha(a.protected_reference_volume)==a.protected_reference_sha256
  with np.load(a.protected_reference_volume) as z:
   reference_p=z['points_world_m'];reference_t=z['tetrahedra']
  assert np.array_equal(reference_p.view(np.uint32),p.view(np.uint32))
  for path,pin in a.protected_cell_map:
   assert sha(path)==pin
   with np.load(path) as z:
    if 'comp0_full_protected_cell_rows' in z:
     rows=z['comp0_full_protected_cell_rows'];pts=z['comp0_full_protected_point_rows']
    elif 'input_cell_rows' in z:
     rows=z['input_cell_rows'];pts=z['global_point_rows']
    else:raise ValueError('unrecognized protected-cell map')
   assert rows.min()>=0 and rows.max()<len(reference_t) and pts.min()>=0 and pts.max()<len(p)
   footprint=np.unique(np.r_[pts,reference_t[rows].ravel()]);fixed[footprint]=True
   protection_records.append({'path':str(path),'sha256':pin,'reference_cell_rows':len(rows),'protected_corner_points':len(footprint),'reference_volume':str(a.protected_reference_volume),'reference_sha256':a.protected_reference_sha256})"""
assert s.count(a)==1;s=s.replace(a,b)
s=s.replace("'registration_map':str(registration),","'protected_cell_maps':protection_records,'all_boundary_coordinates_bit_identical':True,'registration_map':str(registration),")
s=s.replace("ap.add_argument('--seconds',type=float,default=45);","ap.add_argument('--seconds',type=float,default=45);ap.add_argument('--protected-cell-map',nargs=2,action='append',default=[]);ap.add_argument('--protected-reference-volume',type=Path);ap.add_argument('--protected-reference-sha256');")
s=s.replace('fixed complete surface','fixed complete actual boundary (target mismatch remains explicit)')
out=H/'condition_fixed_fat_interior_LP_486_root.py';assert not out.exists();out.write_text(s);py_compile.compile(str(out),doraise=True)

v=(H/'verify_fixed_skin_interior_LP_456_root.py').read_text()
v=v.replace('import json,hashlib,importlib.util,argparse,numpy as np,time','import json,hashlib,importlib.util,argparse,numpy as np,time,csv')
v=v.replace("a=argparse.ArgumentParser();",fn+"\na=argparse.ArgumentParser();")
v=v.replace("H/'combined-child-handoff-142/skin.node'","H/'combined-child-handoff-142/fat.node'")
v=v.replace("reg,counts=R.registered_source_rows_from_map(Path(r['registration_map']),sp);fixed[:len(sp)]|=reg","""reg,counts=registered_fat_source_rows(Path(r['registration_map']),sp);fixed[:len(sp)]|=reg
for protection in r['protected_cell_maps']:
 assert sha(protection['path'])==protection['sha256'] and sha(protection['reference_volume'])==protection['reference_sha256']
 with np.load(protection['reference_volume']) as z:assert np.array_equal(z['points_world_m'].view(np.uint32),p.view(np.uint32));reference_t=z['tetrahedra']
 with np.load(protection['path']) as z:
  if 'comp0_full_protected_cell_rows' in z:rows=z['comp0_full_protected_cell_rows'];pts=z['comp0_full_protected_point_rows']
  else:rows=z['input_cell_rows'];pts=z['global_point_rows']
 footprint=np.unique(np.r_[pts,reference_t[rows].ravel()]);assert len(footprint)==protection['protected_corner_points'];fixed[footprint]=True
assert np.array_equal(p[:len(sp)][reg].view(np.uint32),sp[reg].view(np.uint32))""")
v=v.replace("assert np.array_equal(T.boundary_identity(boundary),T.boundary_identity(new['desired_boundary_faces']))","assert np.array_equal(T.boundary_identity(boundary),T.boundary_identity(T.boundary(new['tetrahedra'])))\nassert np.array_equal(old['desired_boundary_faces'],new['desired_boundary_faces'])")
v=v.replace("'complete_boundary_source_geometry_material_connectivity_preserved':True","'actual_boundary_source_geometry_material_connectivity_preserved':True,'complete_target_boundary':'NOT_NEWLY_QUALIFIED_INHERITED_RESIDUAL_UNCHANGED','protected_cell_maps':r['protected_cell_maps']")
vp=H/'verify_fixed_fat_LP_487_root.py';assert not vp.exists();vp.write_text(v);py_compile.compile(str(vp),doraise=True)
receipt={'status':'PREPARED_FAT_FIXED_BOUNDARY_LP_AND_INDEPENDENT_REPLAY','construction':str(out),'construction_sha256':sha(out),'verifier':str(vp),'verifier_sha256':sha(vp),'parent_constructor_sha256':sha(H/'condition_fixed_skin_interior_LP_455_root.py'),'parent_verifier_sha256':sha(H/'verify_fixed_skin_interior_LP_456_root.py'),'registration_parent_sha256':sha(H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py'),'physical_admission':False,'native_steps':0}
(H/'fat-fixed-boundary-LP-485-preparation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
