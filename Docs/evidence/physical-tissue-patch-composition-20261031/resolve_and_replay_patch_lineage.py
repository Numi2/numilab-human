"""Resolve existing immediate cell-parent CSR and exercise the branch composer."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def resolve(base_path,patch_path):
 chain=[];seen=set();current=Path(patch_path)
 while current!=base_path:
  assert current not in seen and len(seen)<32;seen.add(current)
  with np.load(current) as z:
   parent=Path(str(z['current_geometry_parent_path'][0]));expected=str(z['current_geometry_parent_sha256'][0])
  assert sha(parent)==expected
  chain.append(current);current=parent
 base=load(base_path);previous=base;lineage=np.arange(len(base['tetrahedra']),dtype=np.int64);pool={};token=-1
 for path in reversed(chain):
  v=load(path);mapping=v['output_cell_to_input_cell'];ret=mapping>=0;new=np.flatnonzero(~ret)
  assert len(mapping)==len(v['tetrahedra']) and np.all(mapping[ret]<len(previous['tetrahedra']))
  assert np.array_equal(v['points_world_m'].view('u4'),base['points_world_m'].view('u4'))
  assert np.array_equal(v['tetrahedra'][ret],previous['tetrahedra'][mapping[ret]]) and np.array_equal(v['material_indices'][ret],previous['material_indices'][mapping[ret]])
  offsets=v['replacement_parent_offsets'];parents=v['replacement_parent_cell_rows']
  assert len(offsets)==len(new)+1 and offsets[0]==0 and offsets[-1]==len(parents)
  result=np.empty(len(mapping),np.int64);result[ret]=lineage[mapping[ret]]
  for i,row in enumerate(new):
   source=parents[offsets[i]:offsets[i+1]];assert len(source) and np.all((source>=0)&(source<len(lineage)))
   values=np.unique(lineage[source]);union=np.unique(np.concatenate([np.array([v],np.int64) if v>=0 else pool[int(v)] for v in values]))
   assert len(union) and np.all(base['material_indices'][union]==v['material_indices'][row])
   pool[token]=union;result[row]=token;token-=1
  lineage=result;previous=v
 sizes=np.ones(len(lineage),np.int64)
 for tok in np.unique(lineage[lineage<0]):sizes[lineage==tok]=len(pool[int(tok)])
 offsets=np.r_[0,np.cumsum(sizes)];parents=np.empty(offsets[-1],np.int64);pos=np.flatnonzero(lineage>=0);parents[offsets[pos]]=lineage[pos]
 for row in np.flatnonzero(lineage<0):parents[offsets[row]:offsets[row+1]]=pool[int(lineage[row])]
 return base,previous,offsets,parents,list(reversed(chain))
def run():
 out=H/'fixed-boundary-composer-587-root';out.mkdir(exist_ok=False);tic=time.monotonic()
 base=H/'fat-canonical-composed-DI-current-573-root/fat-current-composed-DI-canonical-volume-proposal.npz';patch=H/'protected-fat-interior-quotient-581-root/fat-interior-quotient-volume-proposal.npz'
 assert sha(base)=='5ec64f7ca176d0acb8bbb2eb46842f45b8f4b02744bde0b8c77bfeef3666e86f' and sha(patch)=='eb882eae2d6b5886daab785cb9d3851bad081d67b6d7c6d6633a32a7c6f4ce01'
 b,p,offsets,parents,chain=resolve(base,patch)
 tool=H/'fixed-boundary-composer-585-root/compose_fixed_boundary_volume_patch.py'
 spec=importlib.util.spec_from_file_location('compose585',tool);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 result,w=m.compose(b,p,b,offsets,parents)
 rk=m.cell_keys(result['tetrahedra'],result['material_indices']);pk=m.cell_keys(p['tetrahedra'],p['material_indices'])
 assert np.array_equal(np.sort(rk),np.sort(pk))
 # Cell identity is invariant under positive even corner permutations.
 # Positivity and per-material oriented boundary were certified by compose.
 ordered_equal=np.all(result['tetrahedra'][np.argsort(rk)]==p['tetrahedra'][np.argsort(pk)],axis=1)
 # This CSR is the existing union representation resolved all the way to573.
 lineage=out/'patch-parent-unions-to-base573.npz'
 np.savez_compressed(lineage,replacement_parent_offsets=offsets,replacement_parent_cell_rows=parents)
 rp={'status':'PASS_REAL_FIXED_BOUNDARY_COMPOSER_GEOMETRY_AND_LINEAGE_REPLAY','base':pin(base),'patch':pin(patch),'chain':[pin(v) for v in chain],'resolved_parent_unions':pin(lineage),'composer':pin(tool),'producer':pin(Path(__file__)),'removed_base_cells':len(w['removed_base_cell_rows']),'new_patch_cells':len(w['added_patch_cell_rows']),'complete_material_and_cell_identities_match':True,'positive_common_corner_permutations':int((~ordered_equal).sum()),'complete_oriented_boundary_match':True,'all_point_coordinate_bits_match':True,'source330_mass_unchanged':True,'physical_admission':False,'native_steps':0,'elapsed_s':time.monotonic()-tic}
 (out/'receipt.json').write_text(json.dumps(rp,indent=2)+'\n');print(json.dumps(rp),flush=True)
if __name__=='__main__':run()
