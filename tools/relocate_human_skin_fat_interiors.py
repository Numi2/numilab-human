#!/usr/bin/env python3
"""Improve free interior skin/fat nodes within their exact fixed cavity boundary.

Preparation only: existing NPZ geometry, full source ancestry and source-mass
compiler remain authoritative. No physical-step code or visual repair.
"""
from pathlib import Path
import argparse,collections,hashlib,json,time,shutil,warnings
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
import simplify_human_skin_fat_boundary as mesh
from build_human_conditioned_cavities import spectral_condition
from rebuild_human_interface_caps import exact_determinants


def chebyshev_point(points,cells,node,maximum_move_m):
 """Maximise minimum opposite-face altitude in a bounded visibility kernel.

 Signed determinants are affine in one vertex. Normalisation keeps the LP
 tolerance independent of world coordinates and tiny tetrahedral volumes.
 Full emitted-F32 quality and exact geometry checks are separate.
 """
 p=np.asarray(points,float);t=np.asarray(cells,np.int64)
 if not len(t) or not np.isfinite(maximum_move_m) or maximum_move_m<=0:raise ValueError('invalid relocation domain')
 hit=t==node
 if not np.all(hit.sum(1)==1):raise ValueError('not a complete one-node star')
 x=p[t];e=x[:,1:]-x[:,:1]
 det=np.einsum('ij,ij->i',np.cross(e[:,0],e[:,1]),e[:,2])
 if not np.all(np.isfinite(x)) or np.any(det<=0):raise ValueError('nonpositive or nonfinite source star')
 g1=np.cross(e[:,1],e[:,2]);g2=np.cross(e[:,2],e[:,0]);g3=np.cross(e[:,0],e[:,1])
 gradient=np.stack([-g1-g2-g3,g1,g2,g3],axis=1)[np.arange(len(t)),hit.argmax(1)]
 norm=np.linalg.norm(gradient,axis=1)
 scale=float(np.max(np.linalg.norm(x-p[node],axis=2)))
 if scale<=0 or np.any(norm<=0):raise ValueError('degenerate opposite faces')
 normal=gradient/norm[:,None];height=det/norm/scale
 radius=maximum_move_m/np.sqrt(3)/scale
 # HiGHS accepts threads=1; SciPy forwards it but warns that it is not
 # part of its own options schema. Suppress that specific forwarding notice.
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',message="Unrecognized options detected:.*threads.*",category=OptimizeWarning)
  result=linprog([0.,0,0,-1.],A_ub=np.column_stack([-normal,np.ones(len(t))]),b_ub=height,
   bounds=[(-radius,radius)]*3+[(0.,None)],method='highs',options={'threads':1})
 if not result.success or not np.isfinite(result.x).all():raise ValueError('bounded altitude LP failed')
 return (p[node]+result.x[:3]*scale).astype(np.float32)


def exact_star_volume_preserved(old_points,new_points,cells):
 """Compare exact determinants on one common coordinate lattice."""
 ids,inverse=np.unique(cells,return_inverse=True);local=inverse.reshape(np.shape(cells))
 both=np.concatenate([old_points[ids],new_points[ids]])
 determinants=exact_determinants(both,np.concatenate([local,local+len(ids)]))
 old=determinants[:len(cells)];new=determinants[len(cells):]
 return bool(min(old)>0 and min(new)>0 and sum(old)==sum(new))


def read_mass(path,count):
 rows=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
 if rows.shape!=(count,2) or not np.array_equal(rows[:,0],np.arange(count)) or not np.isfinite(rows).all() or np.any(rows[:,1]<=0):raise ValueError('invalid source mass domain')
 return rows[:,1].copy()


def uniform_star_mass(points,cells,new_points,old_mass,material):
 old_volume=mesh.measure(points,cells)[0]/6;new_volume=mesh.measure(new_points,cells)[0]/6
 density=np.asarray(old_mass)/old_volume
 if not np.all(np.isfinite(density)) or np.ptp(density)>1e-9*float(density.mean()):raise ValueError('nonuniform density requires general conservative compiler routing')
 total=float(np.sum(old_mass));rho=total/float(old_volume.sum());lo,hi={15:(900.,1100.),16:(850.,950.)}[int(material)]
 if rho<lo-1e-8 or rho>hi+1e-8:raise ValueError('source density outside declared interval')
 mass=new_volume*(total/float(new_volume.sum()));pivot=int(np.argmax(new_volume));mass[pivot]=total-float(np.delete(mass,pivot).sum())
 actual=mass/new_volume
 if np.any(mass<=0) or not np.isfinite(mass).all() or np.any(actual<lo-1e-8) or np.any(actual>hi+1e-8):raise ValueError('replacement density outside declared interval')
 before=np.einsum('i,ij->j',old_mass,np.asarray(points)[cells].mean(1));after=np.einsum('i,ij->j',mass,np.asarray(new_points)[cells].mean(1))
 if np.linalg.norm(after-before)>1e-11*max(total,1e-15):raise ValueError('uniform cavity first moment changed')
 return mass


def verify_relocations(source,candidate,operations,source_mass,candidate_mass,typed_nodes):
 """Independently replay disjoint complete stars; reject undeclared changes."""
 p=source['points_reference_world_m_f64'];new=candidate['points_reference_world_m_f64'];t=source['volume_tetrahedra']
 if p.shape!=new.shape or not np.isfinite(new).all() or not np.array_equal(new,new.astype(np.float32).astype(float)):raise ValueError('invalid emitted relocation points')
 for key in ['volume_tetrahedra','material_index','geometric_donor_body','anatomical_region_id']:
  if not np.array_equal(source[key],candidate[key]):raise ValueError('relocation changed topology or anatomical labels')
 labels=np.column_stack([source[k] for k in ['material_index','geometric_donor_body','anatomical_region_id']])
 fixed=np.zeros(len(p),bool);fixed[mesh.volume_boundary_nodes(t,labels[:,0])]=True;fixed[np.asarray(typed_nodes,dtype=np.int64)]=True
 incident=mesh.incidence(t,len(p));occupied=np.zeros(len(t),bool);seen=set();changed=[];max_move=0.;worst_moment=0.
 if not operations:raise ValueError('missing relocation operations')
 for op in operations:
  node=int(op['node']);rows=np.asarray(op['source_cell_rows'],dtype=np.int64)
  if not 0<=node<len(p) or fixed[node] or node in seen:raise ValueError('relocation touches pinned or repeated node')
  actual=incident(node)
  if not np.array_equal(rows,actual) or occupied[rows].any():raise ValueError('incomplete or dependent relocation star')
  if not np.array_equal(p[node],op['old_world_m']) or not np.array_equal(new[node],op['new_world_m']):raise ValueError('relocation coordinate replay mismatch')
  if not np.all(labels[rows]==labels[rows[0]]) or not np.array_equal(labels[rows[0]],op['labels']):raise ValueError('relocation crosses typed region')
  move=float(np.linalg.norm(new[node]-p[node]));max_move=max(move,max_move)
  if move<=0 or move>.0005:raise ValueError('relocation displacement outside bound')
  cells=t[rows]
  if not exact_star_volume_preserved(p,new,cells):raise ValueError('relocation exact volume or orientation failed')
  oldq=mesh.measure(p,cells)[1];det,nq=mesh.measure(new,cells)
  if np.any(det/6<=1e-18) or nq.min()<max(1e-5,float(oldq.min())*1.05) or spectral_condition(new,cells).max()>.98*spectral_condition(p,cells).max():raise ValueError('relocation quality regressed')
  expected=uniform_star_mass(p,cells,new,source_mass[rows],labels[rows[0],0])
  if not np.array_equal(expected,candidate_mass[rows]):raise ValueError('relocation mass replay mismatch')
  moment=np.linalg.norm(np.einsum('i,ij->j',candidate_mass[rows],new[cells].mean(1))-np.einsum('i,ij->j',source_mass[rows],p[cells].mean(1)))
  worst_moment=max(worst_moment,float(moment));occupied[rows]=True;seen.add(node);changed.append(node)
  sourceoff=source['replacement_parent_offsets'];sourceparents=source['replacement_parent_cell_rows'];off=candidate['replacement_parent_offsets'];parents=candidate['replacement_parent_cell_rows']
  pool=np.unique(np.concatenate([sourceparents[sourceoff[i]:sourceoff[i+1]] for i in rows]))
  for row in rows:
   if not np.array_equal(parents[off[row]:off[row+1]],pool):raise ValueError('relocation ancestry does not cover complete source star')
 if not np.array_equal(np.flatnonzero(np.any(p!=new,axis=1)),np.sort(changed)):raise ValueError('undocumented coordinate change')
 if not np.array_equal(source_mass[~occupied],candidate_mass[~occupied]):raise ValueError('mass changed outside relocation stars')
 # Unchanged rows preserve their exact ancestry segments as well.
 off=candidate['replacement_parent_offsets'];oldoff=source['replacement_parent_offsets'];parents=candidate['replacement_parent_cell_rows'];oldparents=source['replacement_parent_cell_rows']
 if not np.array_equal(np.diff(off)[~occupied],np.diff(oldoff)[~occupied]):raise ValueError('unaffected ancestry counts changed')
 for start in range(0,len(t),65536):
  rows=np.arange(start,min(start+65536,len(t)));rows=rows[~occupied[rows]]
  if not len(rows):continue
  ni=np.concatenate([np.arange(off[i],off[i+1]) for i in rows]);oi=np.concatenate([np.arange(oldoff[i],oldoff[i+1]) for i in rows])
  if not np.array_equal(parents[ni],oldparents[oi]):raise ValueError('unaffected ancestry changed')
 return dict(status='PASS_INDEPENDENT_INTERIOR_RELOCATION_REPLAY',nodes=len(changed),cells=int(occupied.sum()),maximum_displacement_m=max_move,maximum_star_first_moment_error_kg_m=worst_moment,all_typed_and_actual_boundaries_fixed=True,exact_star_volumes_preserved=True,source_mass_and_first_moments_preserved=True,physical_admission=False)


def replay_candidate_admission(directory):
 """Authenticate and replay interior geometry; never admit a flag-only proposal."""
 directory=Path(directory);report_path=directory/'report.json'
 report=json.loads(report_path.read_text())
 required=dict(status='INTERIOR_RELOCATION_PROPOSAL_NOT_PHYSICAL_ADMISSION',physical_admission=False,native_steps=0,native_quality_failures=0,interior_only=True,all_point_coordinates_unchanged=False,typed_surface_coordinates_unchanged=True)
 if any(type(report.get(k)) is not type(v) or report[k]!=v for k,v in required.items()):
  raise ValueError('interior admission requires explicit source-only relocation status')
 pins={}
 def pinned(path,expected=None):
  path=Path(path)
  with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  if expected is not None and digest!=expected:raise ValueError('interior admission input hash mismatch: '+str(path))
  pins[str(path)]=digest;return path
 src=report['relocation_input']
 volume=pinned(src['volume'],src['volume_sha256']);mass=pinned(src['mass'],src['mass_sha256']);original=pinned(src['original'],src['original_sha256'])
 candidate_path=pinned(directory/'candidate-volume.npz');candidate_mass=pinned(directory/'local-child-mass-proposal.csv');pinned(report_path);pinned(__file__)
 with np.load(volume,allow_pickle=False) as z:source={k:z[k] for k in z.files}
 with np.load(candidate_path,allow_pickle=False) as z:candidate={k:z[k] for k in z.files}
 if str(source['source_volume_sha256'].item())!=src['original_sha256']:raise ValueError('interior admission original ancestry pin mismatch')
 with np.load(original,allow_pickle=False) as z:
  typed=np.unique(np.concatenate([z['current_typed_role_source_row_to_joint_node_'+role] for role in 'OID']));typed=typed[typed>=0]
  if not np.array_equal(source['points_reference_world_m_f64'],z['points_reference_world_m_f64'].astype(np.float32).astype(float)):raise ValueError('interior admission source point domain mismatch')
 result=verify_relocations(source,candidate,report['interior_relocations'],read_mass(mass,len(source['volume_tetrahedra'])),read_mass(candidate_mass,len(candidate['volume_tetrahedra'])),typed)
 result['input_sha256']=pins
 return result


def run(a):
 start=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 def sha(p):
  with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 for p,s in [(a.volume,a.volume_sha256),(a.mass_csv,a.mass_sha256),(a.original,a.original_sha256)]:
  if sha(p)!=s:raise ValueError('source pin mismatch: '+str(p))
 shutil.copy2(__file__,a.output/'executed-runner.py')
 with np.load(a.volume) as z:raw={k:z[k] for k in z.files}
 p=raw['points_reference_world_m_f64'];t=raw['volume_tetrahedra'];labels=np.column_stack([raw[k] for k in ['material_index','geometric_donor_body','anatomical_region_id']])
 if not np.array_equal(p,p.astype(np.float32).astype(float)):raise ValueError('source is not emitted Float32')
 sourcep=p.copy();mass=read_mass(a.mass_csv,len(t));source_mass=mass.copy()
 if mass.shape!=(len(t),) or not np.all(np.isfinite(mass)&(mass>0)):raise ValueError('invalid source mass domain')
 fixed=np.zeros(len(p),bool);fixed[mesh.volume_boundary_nodes(t,labels[:,0])]=True
 with np.load(a.original) as z:
  if str(raw['source_volume_sha256'].item())!=a.original_sha256:raise ValueError('original ancestry pin mismatch')
  for role in 'OID':
   ids=z['current_typed_role_source_row_to_joint_node_'+role];fixed[ids[ids>=0]]=True
 incident=mesh.incidence(t,len(p));q=np.empty(len(t))
 for lo in range(0,len(t),8192):q[lo:lo+8192]=mesh.measure(p,t[lo:lo+8192])[1]
 low_quality_before=int(np.count_nonzero(q<.001))
 candidates=np.unique(t[q<.001]);candidates=candidates[~fixed[candidates]]
 candidates=sorted(map(int,candidates),key=lambda n:(float(q[incident(n)].min()),n))
 occupied=np.zeros(len(t),bool);rejected=collections.Counter();operations=[];parents={};off=raw['replacement_parent_offsets'];pr=raw['replacement_parent_cell_rows']
 for node in candidates:
  if time.monotonic()-start>a.seconds or len(operations)>=a.maximum_nodes:break
  rows=incident(node)
  if len(rows)>256 or occupied[rows].any():rejected['dependent_or_large_star']+=1;continue
  if not np.all(labels[rows]==labels[rows[0]]):rejected['typed_region_boundary']+=1;continue
  cells=t[rows];density=mass[rows]/(mesh.measure(p,cells)[0]/6)
  if np.ptp(density)>1e-9*float(density.mean()):rejected['nonuniform_source_density']+=1;continue
  try:proposal=chebyshev_point(p,cells,node,a.maximum_move_m)
  except ValueError as x:rejected[str(x)]+=1;continue
  old=p[node].copy();oldq=q[rows];old_condition=spectral_condition(p,cells);success=False
  for k in range(12):
   new=(old+(proposal.astype(float)-old)*2.**(-k)).astype(np.float32).astype(float)
   if np.array_equal(new,old) or np.linalg.norm(new-old)>a.maximum_move_m:continue
   p[node]=new;det,nq=mesh.measure(p,cells);condition=spectral_condition(p,cells)
   if np.any(det/6<=1e-18) or nq.min()<max(1e-5,float(oldq.min())*1.05) or condition.max()>.98*old_condition.max():continue
   if not exact_star_volume_preserved(sourcep,p,cells):continue
   try:nm=uniform_star_mass(sourcep,cells,p,source_mass[rows],labels[rows[0],0])
   except ValueError:continue
   success=True;break
  if not success:p[node]=old;rejected['no_admissible_emitted_F32_progress']+=1;continue
  mass[rows]=nm;q[rows]=nq;occupied[rows]=True
  pool=np.unique(np.concatenate([pr[off[i]:off[i+1]] for i in rows]))
  for row in rows:parents[int(row)]=pool
  operations.append(dict(node=int(node),old_world_m=old.tolist(),new_world_m=p[node].tolist(),source_cell_rows=rows.tolist(),labels=labels[rows[0]].tolist(),minimum_quality_before=float(oldq.min()),minimum_quality_after=float(nq.min()),maximum_condition_before=float(old_condition.max()),maximum_condition_after=float(condition.max()),mass_before_kg=float(source_mass[rows].sum()),mass_after_kg=float(mass[rows].sum()),exact_star_volume_unchanged=True))
  if len(operations)%128==0:print('passing_interior_nodes',len(operations),'elapsed_s',time.monotonic()-start,flush=True)
 if not operations:raise ValueError('no admissible interior relocation')
 assert np.array_equal(p[fixed],sourcep[fixed])
 counts=np.diff(off).copy()
 for row,pool in parents.items():counts[row]=len(pool)
 output_parents=[]
 for row in range(len(t)):output_parents.append(parents.get(row,pr[off[row]:off[row+1]]))
 raw['replacement_parent_offsets']=np.r_[0,np.cumsum(counts)];raw['replacement_parent_cell_rows']=np.concatenate(output_parents)
 np.savez_compressed(a.output/'candidate-volume.npz',**raw)
 np.savetxt(a.output/'local-child-mass-proposal.csv',np.column_stack([np.arange(len(t)),mass]),delimiter=',',header='child_row,mass_kg',comments='',fmt=['%d','%.17g'])
 source_report=json.loads((a.volume.parent/'report.json').read_text());source_report.update(status='INTERIOR_RELOCATION_PROPOSAL_NOT_PHYSICAL_ADMISSION',physical_admission=False,native_steps=0,interior_only=True,all_point_coordinates_unchanged=False,typed_surface_coordinates_unchanged=True,interior_relocations=operations,relocation_input=dict(volume=str(a.volume),volume_sha256=a.volume_sha256,mass=str(a.mass_csv),mass_sha256=a.mass_sha256,original=str(a.original),original_sha256=a.original_sha256),rejections=dict(rejected),low_quality_cells_before=low_quality_before,low_quality_cells_after=int(np.count_nonzero(q<.001)),eligible_free_interior_nodes=len(candidates),elapsed_s=time.monotonic()-start,local_mass_construction='Complete-star uniform density, per-star mass and first moment conserved; existing source carrier compiler and native admission still mandatory')
 (a.output/'report.json').write_text(json.dumps(source_report,indent=2)+'\n')
 for role in 'OID':(a.output/f'candidate-{role}-typed-current.npz').symlink_to((a.volume.parent/f'candidate-{role}-typed-current.npz').resolve())
 shutil.copy2(a.volume.parent/'context-config.json',a.output/'context-config.json')
 print(json.dumps({k:source_report[k] for k in ['status','eligible_free_interior_nodes','low_quality_cells_before','low_quality_cells_after','elapsed_s','rejections']},indent=2),flush=True)
 print('relocated_nodes',len(operations),flush=True)


if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 for k in ['volume','mass-csv','original','output']:ap.add_argument('--'+k,type=Path,required=True)
 for k in ['volume','mass','original']:ap.add_argument('--'+k+'-sha256',required=True)
 ap.add_argument('--maximum-nodes',type=int,default=2048);ap.add_argument('--maximum-move-m',type=float,default=.0005);ap.add_argument('--seconds',type=float,default=180)
 a=ap.parse_args()
 if a.maximum_nodes<1 or not 0<a.maximum_move_m<=.0005 or a.seconds<=0:raise ValueError('invalid bounded relocation settings')
 run(a)
