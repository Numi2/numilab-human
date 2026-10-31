"""Compose already certified disjoint fixed-boundary patches onto I50/D264."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time,gc
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def run():
 out=H/'current-paired-fixed-boundary-composition-597-root';out.mkdir(exist_ok=False);tic=time.monotonic()
 C=imp('composer585',H/'fixed-boundary-composer-585-root/compose_fixed_boundary_volume_patch.py')
 R=imp('resolve587',H/'resolve_and_verify_fixed_boundary_composition_587_root.py')
 Q=imp('quality198',H/'fixed_boundary_volume_flip_198_root.py')
 canonical=H/'paired-I-local-cavity-527-root/run055-current-I50-D264-source-I-baseline-context/canonical-parent-unions-v3'
 cfg=[
 ('Skin',H/'skin-canonical-paired-I-current-571-root/skin-current-paired-I-canonical-volume-proposal.npz',H/'skin-fixed-boundary-disjoint-I50-584-root/round001/fat-protected-polygon-connectivity-volume-proposal.npz',canonical/'skin/skin-current-I50-canonical-parent-unions.npz','5e2c8354fcd8bc5f27d2557e8f9da822f994bf6c06ce77c653146312a0d7b091',893,890),
 ('Fat',H/'fat-canonical-composed-DI-current-573-root/fat-current-composed-DI-canonical-volume-proposal.npz',H/'protected-fat-interior-quotient-581-root/fat-interior-quotient-volume-proposal.npz',canonical/'fat/fat-current-I50-D264-canonical-parent-unions.npz','4798310f9fc4257ef27618d5323ff1350b1df334ab3f59680665aa8beee4e396',10872,10594)]
 roles={}
 for role,bp,pp,cp,ch,before,after in cfg:
  assert sha(cp)==ch
  b,p,offsets,parents,chain=R.resolve(bp,pp);c=load(cp)
  result,w=C.compose(b,p,c,offsets,parents)
  result.update(current_geometry_parent_path=np.asarray([str(cp)]),current_geometry_parent_sha256=np.asarray([sha(cp)]),
   input_node_to_output_node=np.arange(len(result['points_world_m']),dtype=np.int64),output_node_to_input_node=np.arange(len(result['points_world_m']),dtype=np.int64),
   canonical_parent_union_operation_order=np.asarray('I50_then_D264_then_disjoint_fixed_boundary' if role=='Fat' else 'I50_then_disjoint_fixed_boundary'))
  det,q=Q.measurements(result['points_world_m'],result['tetrahedra']);good=Q.passes(det,q)
  oldd,oldq=Q.measurements(c['points_world_m'],c['tetrahedra'])
  assert int((~Q.passes(oldd,oldq)).sum())==before
  assert np.all(det>0) and int((~good).sum())==after
  target=C.oriented_boundary(result['tetrahedra'])
  # Native target is outward triangles; compare sorted triples plus parity.
  f=result['desired_boundary_faces'];s=np.sort(f,axis=1);sign=1-2*(((f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8))%2)
  order=np.lexsort((s[:,2],s[:,1],s[:,0]));desired=np.c_[s[order],sign[order]]
  assert np.array_equal(target,desired)
  dest=out/(role.lower()+'-current-connected-volume-proposal.npz');np.savez_compressed(dest,**result)
  with np.load(dest) as z:
   assert all(np.array_equal(z[k],v) for k,v in result.items())
  witness=out/(role.lower()+'-composition-witnesses.npz');np.savez_compressed(witness,**w)
  r={'base':pin(bp),'fixed_boundary_patch':pin(pp),'current_before_composition':pin(cp),'candidate':pin(dest),'witnesses':pin(witness),'fixed_boundary_chain':[pin(v) for v in chain],
   'native_quality_before':before,'native_quality_after':after,'nonpositive':int((det<=0).sum()),'cells':len(result['tetrahedra']),'nodes':len(result['points_world_m']),
   'removed_base_cells':len(w['removed_base_cell_rows']),'new_patch_cells':len(w['added_patch_cell_rows']),
   'boundary_faces':len(target),'complete_oriented_target_match':True,'all_coordinate_bits_unchanged':True,
   'immediate_parent_unions_regenerated':True,'mass_mapping':'CURRENT_SOURCE330_ALLOCATION_REQUIRED','physical_admission':False,'native_steps':0}
  if role=='Fat':
   protected=H/'paired-I-local-cavity-527-root/run037-current-D390-I-joint354/protected-comp0-comp1-census-001/protected-current-cells-and-bad-witnesses.npz'
   with np.load(protected) as z:keys=z['source_cell_keys']
   bk=C.cell_keys(b['tetrahedra'],b['material_indices']);rk=C.cell_keys(result['tetrahedra'],result['material_indices'])
   removed=w['removed_base_cell_rows'];added=w['added_patch_cell_rows']
   assert np.all(np.isin(bk[removed],keys))
   newkeys=C.cell_keys(p['tetrahedra'][added],p['material_indices'][added])
   want=np.r_[keys[~np.isin(keys,bk[removed])],newkeys];assert len(np.unique(want))==len(want)==35601
   rows=np.flatnonzero(np.isin(rk,want));assert len(rows)==35601 and int((~good[rows]).sum())==2191
   side=out/'protected-current-cells-and-bad-witnesses.npz'
   np.savez_compressed(side,current_output_cell_rows=rows,source_cell_keys=rk[rows],protected_cell_is_bad=~good[rows],protected_point_rows=np.unique(result['tetrahedra'][rows]))
   r['protected_construction_region']={'sidecar':pin(side),'cells':len(rows),'native_quality_failures':int((~good[rows]).sum()),'real_anatomical_anchor':False}
  roles[role]=r
  (out/'progress.json').write_text(json.dumps(roles,indent=2)+'\n')
  print(json.dumps({'role':role,'candidate':r['candidate'],'bad':after,'cells':r['cells']}),flush=True)
  del b,p,c,result,offsets,parents,det,q,good,oldd,oldq;gc.collect()
 receipt={'status':'PASS_COMBINED_CURRENT_PAIRED_FIXED_BOUNDARY_GEOMETRY_AND_LINEAGE','roles':roles,'producer':pin(Path(__file__)),
 'composer':pin(H/'fixed-boundary-composer-585-root/compose_fixed_boundary_volume_patch.py'),'resolver':pin(H/'resolve_and_verify_fixed_boundary_composition_587_root.py'),
 'canonical_replay':pin(canonical/'receipt.json'),'source_context':pin(canonical.parent/'audit354/report.json'),
 'source330_mass_immutable':True,'native_thresholds_unchanged':True,'surface_crossings':0,'scope':'Reference geometry and canonical immediate-parent lineage, not physical admission.',
 'physical_admission':False,'native_steps':0,'motion_verification':'NOT_RUN','elapsed_s':time.monotonic()-tic}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'status':receipt['status'],'elapsed_s':receipt['elapsed_s'],'receipt':pin(out/'receipt.json')}),flush=True)
if __name__=='__main__':run()
