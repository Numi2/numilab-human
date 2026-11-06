"""Exact bridge from joint mechanical quotient to existing role-volume row domains.
Uses the retained 217 node table. No guessed coordinates, parent rows or mass.
"""
from pathlib import Path
import numpy as np,csv,gzip,json,sys,hashlib,importlib.util
from human_stable_cell_order import replacement_in_reference_order
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
d=Path(sys.argv[1]);out=d/"role-volumes-v2";out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=H/"current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz"
with np.load(source) as z:p=z["points_reference_world_m_f64"].astype(np.float32);t=z["volume_tetrahedra"];mat=z["material_index"]
with np.load(d/"candidate-volume.npz") as z:nt=z["volume_tetrahedra"];nm=z["material_index"];off=z["replacement_parent_offsets"];parents=z["replacement_parent_cell_rows"];current_points=z["points_reference_world_m_f64"].astype(np.float32)
relocation_report=json.loads((d/'report.json').read_text())
relocations=relocation_report.get('interior_relocations',[])
if relocations:
 checks=json.loads((d/'interior-relocation-checks.json').read_text())
 assert checks['status']=='PASS_INDEPENDENT_INTERIOR_RELOCATION_REPLAY'
 assert checks['nodes']==len(relocations)
 assert np.array_equal(np.flatnonzero(np.any(p!=current_points,axis=1)),np.sort([int(op['node']) for op in relocations]))
else:assert np.array_equal(p,current_points)
report={"status":"RUNNING","physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","roles":{}}
rp=out/"report.json"
def save():rp.write_text(json.dumps(report,indent=2)+"\n")
save()
bases={}
for role in ["Skin","Fat"]:
 path=H/f"current-native-1024-root/canonical834-current-base/{role.lower()}-canonical-parent-union.npz"
 with np.load(path) as z:pp=z["points_world_m_f32"];tt=z["tetrahedra"]
 used=np.unique(tt);bases[role]={"path":path,"p":pp,"t":tt,"used":used,"to_joint":np.full(len(pp),-1,np.int64)}
node_table=H/"current-role-p1-support-938-1025-active-source-map/actual-001/217-handoff/skin-fat-combined-node-map.csv.gz"
with gzip.open(node_table,"rt",newline="") as f:
 for row in csv.DictReader(f):
  joint=int(row["joint_node_id"])-1
  for partition,key in [(row["source_partition"],"source_tetgen_node_id"),(row["merge_peer_partition"],"merge_peer_tetgen_node_id")]:
   if not partition:continue
   role=partition.capitalize();base=bases[role]
   # 938 preserves canonical source row + 1 as the TetGen node ID;
   # active rows are sparse, not renumbered in ascending active order.
   point=int(row[key])-1
   assert 0<=point<len(base["p"])
   assert np.searchsorted(base["used"],point)<len(base["used"])
   assert base["used"][np.searchsorted(base["used"],point)]==point
   assert base["to_joint"][point] in [-1,joint]
   base["to_joint"][point]=joint
spec=importlib.util.spec_from_file_location("oldtop",H/"transfer_surface_contractions_indexed_379_root.py");top=importlib.util.module_from_spec(spec);spec.loader.exec_module(top)
for role,material in [("Skin",15),("Fat",16)]:
 base=bases[role];forward=base["to_joint"];used=base["used"];assert np.all(forward[used]>=0)
 original_rows=np.flatnonzero(mat==material)
 assert len(original_rows)==len(base["t"])
 assert np.array_equal(np.sort(forward[base["t"]],axis=1),np.sort(t[original_rows],axis=1))
 inv=np.full(len(p),-1,np.int64);inv[forward[used]]=used
 rows=np.flatnonzero(nm==material);tt=inv[nt[rows]];assert np.all(tt>=0)
 role_points=base['p'].copy();role_points[used]=current_points[forward[used]]
 det,q=top.m.measurements(role_points,tt)
 point_map=np.arange(len(role_points),dtype=np.int64);point_pools={}
 for op in relocations:
  node=int(op['node'])
  if inv[node]<0:continue
  point=int(inv[node]);point_map[point]=-1
  joint_pool=np.unique(np.concatenate([parents[off[i]:off[i+1]] for i in op['source_cell_rows']]))
  role_pool=np.searchsorted(original_rows,joint_pool)
  assert np.all(role_pool<len(original_rows)) and np.array_equal(original_rows[role_pool],joint_pool)
  point_pools[point]=role_pool
 point_counts=np.zeros(len(role_points),np.int64)
 for point,pool in point_pools.items():point_counts[point]=len(pool)
 point_offsets=np.r_[0,np.cumsum(point_counts)]
 point_parents=np.concatenate([point_pools[point] for point in sorted(point_pools)]) if point_pools else np.empty(0,np.int64)
 current_mismatch=np.linalg.norm(role_points[used].astype(float)-current_points[forward[used]].astype(float),axis=1)
 mismatch=np.linalg.norm(role_points[used].astype(float)-p[forward[used]].astype(float),axis=1)
 offsets=np.r_[0,np.cumsum(np.diff(off)[rows])]
 cp=np.concatenate([parents[off[i]:off[i+1]] for i in rows]);lookup=np.full(len(mat),-1,np.int64);lookup[original_rows]=np.arange(len(original_rows));cp=lookup[cp];assert np.all(cp>=0)
 # This is a direct role volume proposal. Parent CSR is in its authenticated
 # immediate canonical role-volume cell-row domain, not source330 ancestry.
 target=out/f"{role.lower()}-volume-proposal.npz"
 boundary=top.boundary(tt)
 direct=cp[offsets[:-1]]
 retained_mask=(np.diff(offsets)==1)&np.all(tt==base["t"][direct],axis=1)
 ri=np.flatnonzero(retained_mask);ri=ri[np.argsort(direct[ri],kind="stable")]
 changed=np.flatnonzero(~retained_mask);order=np.r_[ri,changed];ordered=tt[order]
 retained=direct[ri];inverse=np.full(len(base["t"]),-1,np.int64);inverse[retained]=np.arange(len(retained))
 replacement_offsets=np.r_[0,np.cumsum(np.diff(offsets)[changed])]
 replacement_rows=np.concatenate([cp[offsets[i]:offsets[i+1]] for i in changed])
 stable=None
 if relocations:
  source=relocation_report['relocation_input'];reference_report_path=Path(source['volume']).parent/'role-volumes-v2/report.json'
  reference_report=json.loads(reference_report_path.read_text());reference_pin=reference_report['roles'][role]
  reference_path=Path(reference_pin['path'])
  if sha(reference_path)!=reference_pin['sha256']:raise ValueError('stable reference role hash mismatch')
  if sha(source['volume'])!=source['volume_sha256']:raise ValueError('stable reference source hash mismatch')
  with np.load(source['volume']) as z:source_points=z['points_reference_world_m_f64'].astype(np.float32)
  with np.load(reference_path) as z:reference_tets=z['tetrahedra'];reference_points=z['points_world_m_f32']
  if reference_points.shape!=role_points.shape or not np.array_equal(reference_points[used],source_points[forward[used]]):raise ValueError('stable reference point identity mismatch')
  ordered,replacement_offsets,replacement_rows,order=replacement_in_reference_order(tt,offsets,cp,reference_tets)
  # Every output is explicitly a replacement with complete ancestry. Do not call
  # a relocated volume cell retained merely to preserve its numerical row order.
  retained=np.empty(0,np.int64);inverse=np.full(len(base['t']),-1,np.int64)
  changed=order
  stable=dict(reference=reference_pin,reference_report_sha256=sha(reference_report_path),all_cells_explicit_replacements=True,permuted_from_raw_material_rows=int(np.count_nonzero(order!=np.arange(len(order)))),complete_parent_unions_preserved=True,point_identity='unchanged role input row domain; documented derived interior points remain derived')
 np.savez_compressed(target,points_world_m=role_points,points_world_m_f32=role_points,tetrahedra=ordered,material_indices=np.full(len(tt),material,np.int32),boundary_faces=boundary,
  current_geometry_parent_path=np.array(str(base["path"])),current_geometry_parent_sha256=np.array(sha(base["path"])),
  replacement_parent_offsets=replacement_offsets,replacement_parent_cell_rows=replacement_rows,
  retained_input_cell_rows=retained,replacement_tetrahedra=ordered if stable else tt[changed],
  output_cell_to_input_cell=direct[order],input_cell_to_output_cell=inverse,
  output_point_to_input_point_row=point_map,
  output_point_parent_offsets=point_offsets,output_point_parent_input_cell_rows=point_parents)
 report["roles"][role]={"path":str(target),"sha256":sha(target),"cells":len(tt),"active_nodes":len(np.unique(tt)),"nonpositive_cells":int((det<=0).sum()),"native_quality_failures":int((~top.m.passes(det,q)).sum()),"minimum_volume_m3":float(det.min()/6),"minimum_quality":float(q.min()),"relocated_interior_points":len(point_pools),"joint_current_coordinate_difference_max_m":float(current_mismatch.max()),"joint_source_coordinate_difference_rms_m":float(np.sqrt(np.mean(mismatch**2))),"joint_source_coordinate_difference_max_m":float(mismatch.max()),"original_ordered_cell_domains_proven":True,"immediate_parent_references":len(cp),"stable_cell_order":stable}
 save()
report["status"]="SOURCE_ROLE_VOLUME_BRIDGE_CHECKED_NOT_PHYSICAL_ADMISSION";save()
print(json.dumps(report,indent=2))
