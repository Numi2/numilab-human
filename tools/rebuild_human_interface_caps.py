#!/usr/bin/env python3
"""Connected skin/fat interface construction through the existing volume compiler.

Only preparation: remove a flat interface cap with an exact 3-to-2 cavity flip,
update the same inferred interface on both tissues, and retain typed ancestry.
The existing source-mass compiler and complete native scene retain admission.
"""
import itertools
import numpy as np
from simplify_human_skin_fat_boundary import unchanged_cavity_boundary, TET_FACES


def exact_determinants(points, cells):
 """Integer determinants on the exact emitted Float32 coordinate lattice."""
 ids=np.unique(cells);ratios={int(i):[float(v).as_integer_ratio() for v in points[i]] for i in ids}
 power=max(d.bit_length()-1 for row in ratios.values() for n,d in row)
 lattice={i:[n<<(power-(d.bit_length()-1)) for n,d in row] for i,row in ratios.items()}
 out=[]
 for cell in cells:
  a,b,c,d=[lattice[int(i)] for i in cell]
  b=[v-u for u,v in zip(a,b)];c=[v-u for u,v in zip(a,c)];d=[v-u for u,v in zip(a,d)]
  out.append(b[0]*(c[1]*d[2]-c[2]*d[1])-b[1]*(c[0]*d[2]-c[2]*d[0])+b[2]*(c[0]*d[1]-c[1]*d[0]))
 return out


def positive_cells(points,cells):
 cells=np.array(cells,np.int64,copy=True)
 det=exact_determinants(points,cells)
 for i,d in enumerate(det):
  if d==0:raise ValueError("degenerate proposed cap cell")
  if d<0:cells[i,[2,3]]=cells[i,[3,2]]
 return cells


def interface_flip(points,cap,neighbors):
 """Propose the two majority-tissue cells replacing cap + two face neighbors.

 Caller proves the removed edge has exactly this three-cell star and the new
 separator is unused, and supplies material/donor, mass and quality admission.
 No coordinates change. The old/new interface has the same exact boundary.
 """
 p=np.asarray(points)
 cap=np.asarray(cap,np.int64);neighbors=np.asarray(neighbors,np.int64)
 if cap.shape!=(4,) or neighbors.shape!=(2,4) or len(np.unique(cap))!=4:raise ValueError("invalid cap domain")
 if min(cap.min(),neighbors.min())<0 or max(cap.max(),neighbors.max())>=len(p):raise ValueError("cap index outside node table")
 local=p[np.unique(np.r_[cap,neighbors.ravel()])]
 if not np.isfinite(local).all() or not np.array_equal(local.astype(np.float32).astype(float),local.astype(float)):
  raise ValueError("cap coordinates must be exact finite emitted Float32")
 c=set(map(int,cap));oldfaces=[];apices=[]
 for neighbor in neighbors:
  common=c&set(map(int,neighbor));apex=set(map(int,neighbor))-c
  if len(common)!=3 or len(apex)!=1:raise ValueError("neighbor is not across an interface face")
  oldfaces.append(common);apices.append(apex.pop())
 if apices[0]!=apices[1] or oldfaces[0]==oldfaces[1]:raise ValueError("not a three-cell edge star")
 edge=sorted(oldfaces[0]&oldfaces[1]);ring=sorted(c-set(edge))
 if len(edge)!=2 or len(ring)!=2:raise ValueError("invalid interface diagonal")
 apex=apices[0]
 new=positive_cells(p,[[apex,*ring,edge[0]],[apex,*ring,edge[1]]])
 old=np.vstack([cap,neighbors])
 od=exact_determinants(p,old);nd=exact_determinants(p,new)
 if min(od)<=0 or min(nd)<=0 or sum(od)!=sum(nd) or not unchanged_cavity_boundary(old,new):
  raise ValueError("cap flip changes the exact oriented cavity or volume")
 # Both diagonal triangulations must be single-valued over one convex quad.
 x=p[sorted(c)].astype(float);_,_,basis=np.linalg.svd(x-x.mean(0),full_matrices=False)
 q={i:(p[i].astype(float)-x.mean(0))@basis[:2].T for i in c}
 u=q[edge[1]]-q[edge[0]];v=q[ring[1]]-q[ring[0]]
 ts=np.linalg.solve(np.column_stack([u,-v]),q[ring[0]]-q[edge[0]])
 if np.any(ts<=0) or np.any(ts>=1):raise ValueError("interface diagonals do not cross inside the quad")
 gap=np.linalg.norm((p[edge[0]].astype(float)+ts[0]*(p[edge[1]].astype(float)-p[edge[0]]))-(p[ring[0]].astype(float)+ts[1]*(p[ring[1]].astype(float)-p[ring[0]])))
 # Match the orientation of the majority material's outward boundary. The
 # caller retains the old typed interface's inward/outward convention.
 newfaces=[]
 for tet in new:
  faces=[tet[pat] for pat in TET_FACES if set(map(int,tet[pat]))<=c]
  if len(faces)!=1:raise ValueError("new cell lacks one inferred interface face")
  newfaces.append(faces[0])
 return dict(cells=new,old_interface_faces=[sorted(f) for f in oldfaces],new_interface_faces=np.array(newfaces),
  removed_edge=edge,new_separator=sorted([apex,*ring]),interface_hausdorff_bound_m=float(gap),
  coordinates_unchanged=True,exact_oriented_cavity_and_volume=True)


def rewrite_face_unions(payload,changed):
 """Compose the existing typed face-union CSR for both sides of each flip."""
 for offset,lanes in [("quality_star_patch_union_offsets",["quality_star_patch_union_face_source_"+k for k in ["layer","body","bone","face_id","original_id","identity0","identity1"]]),("face_ancestry_offsets",["face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows"])]:
  old=payload[offset];chunks=[]
  for row in range(len(payload['faces'])):
   parents=changed.get(row,[row]);ix=np.concatenate([np.arange(old[i],old[i+1]) for i in parents])
   chunks.append(np.unique(np.column_stack([payload[k][ix] for k in lanes]),axis=0))
  union=np.concatenate(chunks);payload[offset]=np.r_[0,np.cumsum([len(x) for x in chunks])]
  for j,k in enumerate(lanes):payload[k]=union[:,j]


def candidate_context(config,oldroot,output):
 """Bind every emitted role by content; include the changed shared interface."""
 from pathlib import Path
 import json,hashlib
 output=Path(output)
 context=json.loads(json.dumps(config).replace(str(oldroot),str(output.resolve())))
 for role in "OID":
  asset=output/f"candidate-{role}-typed-current.npz"
  context['candidate_surfaces'][role]={'path':str(asset.resolve()),'sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}
 context['changed_roles']=list(dict.fromkeys([*context.get('changed_roles',[]),'I']))
 return context


def registered_joint_to_original_node(original_rows,current_rows,count):
 """Join authenticated 938 canonical role-node IDs; never semantic anchors."""
 source={}
 for row in original_rows:
  joint=int(row['joint_node_id'])-1
  for role,key in [(row['source_partition'],'source_tetgen_node_id'),(row['merge_peer_partition'],'merge_peer_tetgen_node_id')]:
   if not role:continue
   token=(role,int(row[key]))
   if token in source and source[token]!=joint:raise ValueError('ambiguous original role node')
   source[token]=joint
 out=np.full(count,-1,np.int64)
 for row in current_rows:
  joint=int(row['joint_node_id'])-1;parents=[]
  if joint<0 or joint>=count:raise ValueError('invalid current node row')
  for role,key in [(row['source_partition'],'source_tetgen_node_id'),(row['merge_peer_partition'],'merge_peer_tetgen_node_id')]:
   if role:
    token=(role,int(row[key]))
    if token not in source:raise ValueError('new role node absent from original source')
    parents.append(source[token])
  if not parents or len(set(parents))!=1:raise ValueError('merged interface disagrees on original node identity')
  if out[joint]>=0 and out[joint]!=parents[0]:raise ValueError('conflicting repeated current joint identity')
  out[joint]=parents[0]
 if np.any(out<0) or len(np.unique(out))!=count:raise ValueError('incomplete or noninjective joint node ancestry')
 return out


def bounded_patch_mass(old_mass,old_material,new_material,new_volume,preferred):
 """Keep preferred retained masses; move only the residual within material boxes.

 Exact source-carrier transport and native admission remain separate. This
 avoids homogenizing all retained halo cells merely to fund a tiny interface
 volume change. Tolerances match the existing density owner.
 """
 old_mass=np.asarray(old_mass,float);old_material=np.asarray(old_material)
 new_material=np.asarray(new_material);new_volume=np.asarray(new_volume,float);preferred=np.asarray(preferred,float)
 if old_mass.shape!=old_material.shape or preferred.shape!=new_volume.shape or new_material.shape!=new_volume.shape or not np.all(np.isfinite(old_mass)&(old_mass>0)) or not np.all(np.isfinite(new_volume)&(new_volume>0)) or not np.all(np.isfinite(preferred)&(preferred>0)):
  raise ValueError('invalid bounded patch mass domain')
 if set(map(int,old_material))!=set(map(int,new_material)):raise ValueError('patch loses a material')
 out=np.empty(len(new_volume))
 for mat in np.unique(old_material):
  low,high={15:(900.,1100.),16:(850.,950.)}[int(mat)]
  rows=np.flatnonzero(new_material==mat);vol=new_volume[rows];lo=low*vol;hi=high*vol
  total=float(old_mass[old_material==mat].sum());target=np.clip(preferred[rows],lo,hi)
  residual=total-float(target.sum())
  # Changed cells and immediate neighbors precede optional halo donors. Keep
  # later retained values exact whenever the nearer cells have capacity.
  for j in range(len(rows)):
   capacity=hi[j]-target[j] if residual>0 else target[j]-lo[j]
   amount=np.copysign(min(abs(residual),max(0.,float(capacity))),residual)
   target[j]+=amount;residual=total-float(target.sum())
   if residual==0:break
  # Restore only arithmetic roundoff on a large cell, never a mass floor.
  pivot=int(np.argmax(vol));target[pivot]+=total-float(target.sum())
  density=target/vol;tol=64*np.finfo(float).eps*np.maximum.reduce([np.ones(len(rows)),np.full(len(rows),high),np.abs(density)])
  if not np.all(np.isfinite(target)&(target>0)&(density>=low-tol)&(density<=high+tol)) or abs(float(target.sum())-total)>64*np.finfo(float).eps*total:
   raise ValueError('material mass cannot fit unchanged local density bounds')
  out[rows]=target
 return out


def run(a):
 from pathlib import Path
 import json,time,hashlib,collections,shutil
 import simplify_human_skin_fat_boundary as mesh
 from build_human_conditioned_cavities import spectral_condition,validate_mass_proof,sha
 start=time.monotonic();cfg=json.loads(a.config.read_text());pins=cfg['inputs']
 def path(k):return Path(pins[k]['path'])
 for ref in pins.values():
  if sha(ref['path'])!=ref['sha256']:raise ValueError('source input hash mismatch: '+ref['path'])
 a.output.mkdir(exist_ok=False,parents=True);shutil.copy2(__file__,a.output/'executed-runner.py')
 with np.load(path('volume'),allow_pickle=False) as s:raw={k:s[k] for k in s.files}
 p=raw['points_reference_world_m_f64'];t=raw['volume_tetrahedra'];labels=np.column_stack([raw[k] for k in ['material_index','geometric_donor_body','anatomical_region_id']])
 if not np.array_equal(p,p.astype(np.float32).astype(float)):raise ValueError('source is not emitted Float32')
 if str(raw['source_volume_sha256'].item())!=pins['original_volume']['sha256']:raise ValueError('original ancestry identity mismatch')
 with np.load(path('original_volume'),allow_pickle=False) as origin:
  original_labels=np.column_stack([origin[k] for k in ['material_index','geometric_donor_body','anatomical_region_id']])
  if not np.array_equal(origin['points_reference_world_m_f64'],p):raise ValueError('original point table changed')
 off=raw['replacement_parent_offsets'];pr=raw['replacement_parent_cell_rows']
 if off.shape!=(len(t)+1,) or off[0]!=0 or off[-1]!=len(pr) or np.any(np.diff(off)<=0) or np.any(pr<0) or pr.max()>=len(original_labels):raise ValueError('invalid immediate source ancestry')
 for lo in range(0,len(t),65536):
  hi=min(lo+65536,len(t));rows=pr[off[lo]:off[hi]]
  if not np.array_equal(original_labels[rows],np.repeat(labels[lo:hi],np.diff(off[lo:hi+1]),axis=0)):raise ValueError('source ancestry changes typed material/donor/region')

 with np.load(path('registered'),allow_pickle=False) as s:
  ct=s['volume_tetrahedra'];xyz=s['points_reference_world_m_f64']
  cl=np.column_stack([s[k] for k in ['material_index','geometric_donor_body','anatomical_region_id']]);im=s['current_typed_role_source_row_to_joint_node_I']
 import csv,gzip
 with gzip.open(path('original_node_map'),'rt',newline='') as oldmap,gzip.open(path('current_node_map'),'rt',newline='') as newmap:
  cp=registered_joint_to_original_node(csv.DictReader(oldmap),csv.DictReader(newmap),len(xyz))
 if np.any(cp<0) or cp.max()>=len(p) or not np.array_equal(p[cp],xyz):raise ValueError('registered raw node identity mismatch')
 def keys(c):return np.ascontiguousarray(np.sort(c,axis=1),dtype=np.int64).view(np.dtype((np.void,32))).ravel()
 rk=keys(t);ck=keys(cp[ct]);order=np.argsort(rk);pos=np.searchsorted(rk[order],ck)
 if np.any(pos>=len(t)) or not np.array_equal(rk[order[pos]],ck) or len(np.unique(rk))!=len(t):raise ValueError('registered raw cell identity mismatch')
 mapped=order[pos]
 if not np.array_equal(labels[mapped,0],cl[:,0]):raise ValueError('registered material identity changed')
 primary=np.empty_like(labels);primary[mapped]=cl
 differences=labels[mapped]!=cl
 primary_label_diagnostic=dict(scope='Raw immediate ancestry labels and registered primary P1-anchor labels are distinct existing compiler domains; full source ancestry remains authoritative.',
  changed_by_field=dict(zip(['material','geometric_donor','anatomical_region'],map(int,differences.sum(0)))),
  witnesses=[dict(raw_cell=int(mapped[i]),registered_cell=int(i),source_ancestry_labels=labels[mapped[i]].tolist(),registered_primary_labels=cl[i].tolist()) for i in np.flatnonzero(differences.any(1))[:16]])
 mass_rows=np.loadtxt(path('mass_csv'),delimiter=',',skiprows=1,ndmin=2)
 if mass_rows.shape!=(len(ct),2) or not np.array_equal(mass_rows[:,0],np.arange(len(ct))) or np.any(mass_rows[:,1]<=0) or not np.isfinite(mass_rows).all():raise ValueError('invalid actual mass targets')
 validate_mass_proof(json.loads(path('mass_parity').read_text()),json.loads(path('mass_receipt').read_text()),pins['registered']['sha256'],pins['mass_csv']['sha256'],pins['mass_receipt']['sha256'],len(ct))
 mass=np.empty(len(t));mass[mapped]=mass_rows[:,1]
 with np.load(path('I'),allow_pickle=False) as s:inner={k:s[k] for k in s.files}
 im_raw=np.full(len(im),-1,np.int64);valid=im>=0;im_raw[valid]=cp[im[valid]]
 if len(im)!=len(inner['points_world_m_f32']) or not np.array_equal(p[im_raw[valid]],inner['points_world_m_f32'][valid]):raise ValueError('typed inner source node mismatch')
 face=im_raw[inner['faces']]
 if np.any(face<0):raise ValueError('inactive typed interface face')
 lookup={tuple(sorted(map(int,f))):i for i,f in enumerate(face)}
 if len(lookup)!=len(face):raise ValueError('duplicate typed interface face')
 reverse=np.full(len(p),-1,np.int64)
 # Only rows referenced by the existing interface need an inverse; no seam merge.
 for row in np.unique(inner['faces']):
  node=im_raw[row]
  if reverse[node]>=0:raise ValueError('ambiguous active typed interface vertex')
  reverse[node]=row
 is_inner=np.zeros(len(p),bool);is_inner[im_raw[valid]]=True
 q=np.empty(len(t))
 for lo in range(0,len(t),8192):q[lo:lo+8192]=mesh.measure(p,t[lo:lo+8192])[1]
 caps=np.flatnonzero(np.all(is_inner[t],axis=1)&(q<.001))
 eligible_caps=len(caps)
 condition=spectral_condition(p,t[caps]);caps=caps[np.lexsort((caps,-condition))][:a.maximum_caps]
 seeds=[int(mapped[i]) for i in cfg.get('seed_registered_cells',[])]
 caps=np.r_[seeds,[int(i) for i in caps if i not in set(seeds)]].astype(np.int64)
 incident=mesh.incidence(t,len(p))
 def together(nodes):
  rows=incident(int(nodes[0]))
  for n in nodes[1:]:rows=np.intersect1d(rows,incident(int(n)),assume_unique=True)
  return rows
 used=set();retired=set();operations=[];rejected=collections.Counter();changed_faces={};new_cells=[];new_labels=[];new_mass=[];new_parents=[]
 offsets=raw['replacement_parent_offsets'];parents=raw['replacement_parent_cell_rows'];rejection_witnesses=[];tested_caps=0;mass_failure_witnesses=[]
 def reject(reason,row):
  rejected[reason]+=1
  if len(rejection_witnesses)<32 or int(row) in seeds:rejection_witnesses.append(dict(source_cap_row=int(row),registered_witness=int(row) in seeds,reason=reason))
 def sign(f):return 1 if sum(f[i]>f[j] for i in range(3) for j in range(i+1,3))%2==0 else -1
 for caprow in caps:
  if time.monotonic()-start>a.seconds:break
  tested_caps+=1
  cap=t[caprow];neighbors=[]
  for pat in TET_FACES:
   rr=together(cap[pat]);rr=rr[rr!=caprow]
   if len(rr)!=1:break
   neighbors.append(int(rr[0]))
  if len(neighbors)!=4:reject('noninterior_face_star',caprow);continue
  opposite=[r for r in neighbors if labels[r,0]!=labels[caprow,0]]
  if len(opposite)!=2 or labels[opposite[0],0]!=labels[opposite[1],0]:reject('not_two_sided_interface_cap',caprow);continue
  other=[r for r in neighbors if r not in opposite];rows=np.array([caprow,*opposite,*other])
  if any(int(n) in used for n in t[rows].ravel()):reject('dependent_patch',caprow);continue
  if not np.all(labels[rows,1:]==labels[caprow,1:]) or not np.all(primary[rows,1:]==primary[caprow,1:]) or not np.all(labels[other,0]==labels[caprow,0]):reject('typed_region_boundary',caprow);continue
  try:flip=interface_flip(p,cap,t[opposite])
  except (ValueError,np.linalg.LinAlgError):reject('exact_flip_or_quad',caprow);continue
  edge=flip['removed_edge'];core={int(caprow),*opposite}
  if set(map(int,together(edge)))!=core or len(together(flip['new_separator'])):reject('nonlocal_edge_or_existing_separator',caprow);continue
  oldkeys=[tuple(f) for f in flip['old_interface_faces']]
  if not all(k in lookup for k in oldkeys):reject('not_current_typed_interface',caprow);continue
  ir=[lookup[k] for k in oldkeys]
  nf=flip['new_interface_faces'].copy()
  if any(tuple(sorted(map(int,f))) in lookup for f in nf):reject('existing_interface_face',caprow);continue
  if flip['interface_hausdorff_bound_m']>a.maximum_gap_m:reject('interface_gap',caprow);continue
  new=flip['cells'];nd,nq=mesh.measure(p,new)
  if np.any(nd/6<=1e-18) or np.any(nq<np.float32(1e-5)) or nq.min()<=q[[caprow,*opposite]].min():reject('native_quality',caprow);continue
  before=float(spectral_condition(p,t[[caprow,*opposite]]).max());after=float(spectral_condition(p,new).max())
  if after>=.75*before:reject('insufficient_conditioning_gain',caprow);continue
  core_rows=np.array([caprow,*opposite]);rest=list(other);halo=[]
  def mass_proposal():
   combined=np.vstack([new,t[rest]]);nl=np.vstack([np.tile(labels[opposite[0]],(2,1)),labels[rest]])
   volume=mesh.measure(p,combined)[0]/6
   preferred=np.r_[float(mass[opposite].sum())*volume[:2]/volume[:2].sum(),mass[rest]]
   rr=np.r_[core_rows,rest].astype(np.int64)
   return rr,combined,nl,bounded_patch_mass(mass[rr],labels[rr,0],nl[:,0],volume,preferred)
  try:rows,combined,nl,nm=mass_proposal()
  except ValueError:
   # A skin cap can be exactly at the density bound. A bounded one-ring halo
   # supplies only existing mass from the same original and registered region.
   candidates=np.unique(np.concatenate([incident(int(n)) for n in cap]))
   candidates=[int(i) for i in candidates if i not in set(rows) and np.array_equal(labels[i,1:],labels[caprow,1:]) and np.array_equal(primary[i,1:],primary[caprow,1:]) and not any(int(n) in used for n in t[i])]
   candidates.sort(key=lambda i:float(np.linalg.norm(p[t[i]].mean(0)-p[cap].mean(0))))
   success=False
   for i in candidates[:16]:
    rest.append(i);halo.append(i)
    try:rows,combined,nl,nm=mass_proposal();success=True;break
    except ValueError:pass
   if not success:
    if int(caprow) in seeds:
     combined0=np.vstack([new,t[rest]]);newmat0=np.r_[np.repeat(labels[opposite[0],0],2),labels[rest,0]];vv=mesh.measure(p,combined0)[0]/6;rr=np.r_[core_rows,rest].astype(np.int64)
     mass_failure_witnesses.append(dict(source_cap_row=int(caprow),old_rows=rr.tolist(),material_totals={str(int(m)):dict(mass_kg=float(mass[rr][labels[rr,0]==m].sum()),new_volume_m3=float(vv[newmat0==m].sum())) for m in np.unique(newmat0)}))
    reject('material_mass_or_density',caprow);continue
  old_boundary=mesh.oriented_tet_boundary(t[opposite]);factors=[]
  for i in ir:
   f=list(map(int,face[i]));factors.append(sign(f)*old_boundary[tuple(sorted(f))])
  if factors[0]!=factors[1] or abs(factors[0])!=1:reject('typed_orientation',caprow);continue
  if factors[0]<0:nf[:,[1,2]]=nf[:,[2,1]]
  if np.any(reverse[nf]<0):raise ValueError('new interface references an unregistered point')
  for i,f in zip(ir,nf):inner['faces'][i]=reverse[f];changed_faces[i]=ir
  retired.update(map(int,rows));used.update(map(int,t[rows].ravel()))
  pools={int(mat):np.unique(np.concatenate([parents[offsets[i]:offsets[i+1]] for i in rows if labels[i,0]==mat])) for mat in np.unique(labels[rows,0])}
  new_cells.extend(combined);new_labels.extend(nl);new_mass.extend(nm);new_parents.extend([pools[int(mat)] for mat in nl[:,0]])
  operations.append(dict(source_cap_row=int(caprow),source_cell_rows=rows.tolist(),new_cells=combined.tolist(),new_materials=nl[:,0].tolist(),inner_face_rows=ir,interface_gap_bound_m=flip['interface_hausdorff_bound_m'],condition_before=before,condition_after=after,exact_cavity_and_volume=True,material_mass_conserved=True,mass_halo_source_rows=halo))
  if len(operations)%32==0:print('passing_interface_caps',len(operations),flush=True)
 if not operations:
  (a.output/'failure-witnesses.json').write_text(json.dumps(dict(status='NO_ADMISSIBLE_CONNECTED_INTERFACE_REPAIR',rejections=dict(rejected),witnesses=rejection_witnesses,primary_label_domains=primary_label_diagnostic),indent=2)+'\n')
  raise ValueError('no admissible connected interface repairs')
 alive=np.ones(len(t),bool);alive[list(retired)]=False;new_cells=np.asarray(new_cells);new_labels=np.asarray(new_labels)
 raw['volume_tetrahedra']=np.vstack([t[alive],new_cells])
 for j,k in enumerate(['material_index','geometric_donor_body','anatomical_region_id']):raw[k]=np.r_[labels[alive,j],new_labels[:,j]]
 counts=np.r_[np.diff(offsets)[alive],[len(v) for v in new_parents]]
 raw['replacement_parent_offsets']=np.r_[0,np.cumsum(counts)];raw['replacement_parent_cell_rows']=np.r_[parents[np.repeat(alive,np.diff(offsets))],np.concatenate(new_parents)]
 np.savez_compressed(a.output/'candidate-volume.npz',**raw)
 outmass=np.r_[mass[alive],new_mass];np.savetxt(a.output/'local-child-mass-proposal.csv',np.column_stack([np.arange(len(outmass)),outmass]),delimiter=',',header='child_row,mass_kg',comments='',fmt=['%d','%.17g'])
 rewrite_face_unions(inner,changed_faces)
 inner['typed_parent_candidate_path']=np.array(str(path('I')));inner['typed_parent_candidate_sha256']=np.array(pins['I']['sha256'])
 np.savez_compressed(a.output/'candidate-I-typed-current.npz',**inner)
 for role in ['O','D']:(a.output/f'candidate-{role}-typed-current.npz').symlink_to(path(role).resolve())
 report=json.loads(path('source_report').read_text());report['source_report_history']={k:report.pop(k) for k in ['operations','rejections','local_mass_construction'] if k in report}
 # The original quotient remains the exact outer-boundary reference for the
 # independent existing compiler checker; interface operations have no quotient.
 report['operations']=report['source_report_history']['operations']
 report.update(status='CONNECTED_INTERFACE_CAP_PROPOSAL_NOT_PHYSICAL_ADMISSION',physical_admission=False,native_steps=0,interior_only=False,
  interface_operations=operations,interface_inputs=pins,primary_label_domains=primary_label_diagnostic,eligible_interface_caps=eligible_caps,tested_interface_caps=tested_caps,mass_failure_witnesses=mass_failure_witnesses,rejections=dict(rejected),rejection_witnesses=rejection_witnesses,current_input_cells=len(t),output_cells=len(raw['volume_tetrahedra']),low_quality_cells_before=int(np.count_nonzero(q<.001)),low_quality_cells_after=int(np.count_nonzero(q[alive]<.001)+np.count_nonzero(mesh.measure(p,new_cells)[1]<.001)),all_point_coordinates_unchanged=True,
  unchanged_outer_and_deep_asset_hashes={r:pins[r]['sha256'] for r in ['O','D']},elapsed_s=time.monotonic()-start,
  local_mass_construction={'scope':'Per-patch material mass and unchanged density ranges only; source-flow and native admission must be repeated','actual_reference_mass':pins['mass_csv']},
  fidelity_qualification='PENDING_REFERENCE_STRAIN_FORCE_ENERGY_AND_NATIVE_COMPARISON')
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 context=json.loads(path('context_config').read_text());oldroot=str(path('source_report').parent)
 context=candidate_context(context,oldroot,a.output);(a.output/'context-config.json').write_text(json.dumps(context,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['status','current_input_cells','output_cells','elapsed_s','rejections']},indent=2));print('interface_operations',len(operations),flush=True)


if __name__=='__main__':
 from pathlib import Path
 import argparse
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--maximum-caps',type=int,default=4096);ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--maximum-gap-m',type=float,default=.000020)
 a=ap.parse_args()
 if a.maximum_caps<1 or a.seconds<=0 or not 0<a.maximum_gap_m<=.000020:raise ValueError('invalid bounded construction settings')
 run(a)
