#!/usr/bin/env python3
"""Build current-1020 coupled FEM harmonic extension and one bounded 030c I-shape trial."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
BASE=H/'joint-local-i-patch-preflight-002'
STATE=H/'all-quality-1020-zero-skin-fat-source-qualified-root/verified-state.json'
TYPED=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
I_PATH=H/'all-quality-1009-point-update-1018-root/actual-001/candidate-I-point-updated.npz'
O_PATH=H/'all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz'
D_PATH=H/'all-quality-958-D014-root/proposal-003/candidate-D-typed-current.npz'
SKIN_PATH=H/'all-quality-1009-point-update-1018-root/actual-001/skin-volume-point-updated.npz'
FAT_PATH=H/'all-quality-1009-point-update-1018-root/actual-001/fat-volume-point-updated.npz'
INV=BASE/'run-005-root-inventory/sensitivities.npz'
INV_REPORT=BASE/'run-005-root-inventory/report.json'
CENSUS_DIR=H/'current-fractional-semantic-native-1047/actual-030c-component-census-001/run-001'
CENSUS=CENSUS_DIR/'component-census/component-census.json'
VOLUME_BOUNDS=CENSUS_DIR/'component-volume-bounds.json'
TARGET_ALL=H/'current-fractional-semantic-native-1047/actual-030c-component-census-001/target-rows/target-side-child-rows.csv'
OWNER=H/'all-quality-1009-banks-point-update-001/emit_and_replay_point_updates_001.py'
PROTECTION_OWNER=BASE/'derive_current1020_cut_gradients_007.py'
BOUNDARY_REPORT=BASE/'boundary-face-classification-002/report.json'
BOUNDARY_WITNESSES=BASE/'boundary-face-classification-002/boundary-difference-witnesses.npz'
FACET_ANCESTRY=H/'current-native-1024-root/current-433-1044/volume-export/volume/facet-ancestry.csv'
ROLE_MAP_REPORT=BASE/'cell-incidence-role-joint-map-root-002/report.json'
ROLE_MAP_NPZ=BASE/'cell-incidence-role-joint-map-root-002/maps.npz'
HELPER=H/'mass-aware-inner-I-001/harmonic_i_extension_001.py'
OUT=BASE/'run-012-current030c-harmonic-fit-root006'
PINS={'state':'11044a88b2d45ab9a9625f9a2cac46b8a952032b9ac0a07274710c316f5a88a9','typed':'5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372','I':'5557cbb88b46ea83d7977d60e1d362a10c48c290b77afb39b273efa29d32e323','O':'10169962e298586d63794bf94515ecc3a9b1ce91e66cc60566a6919d7c856ac3','D':'63cec05102fd2dc670d9220ec6472cd7d9ffa7085b4694719c605947086c7da9','Skin':'41291a7298517ab250a64e4bbda6bbee7591b7d2b594aacc1dce6d6fc78025ba','Fat':'b83df0f4a4ff183bd147f0c0d6236a22e3ea6ce1ae9e9531ffc1c457c2ffac88','inventory':'a3d1d5c8d4933f9bdd24dda0a015471742398a6e333e7409c80a147f9dad127f','inventory_report':'b9277ea48a59e2e09ee99e66c45a54a29b2fc5336679764add230cc0868adda2','census':'d8ac2770b3fa5bde237aebe20a97a1b28bbae9a5ce1f58bc3406341305f13c62','volume_bounds':'e349b2ff9d67e4a4602d5174b3f0076deba8e84e4b888fe42ba3857c654f41a7','target':'14252ebfe9c5c42adb5f1641dbd3446a5d9db739620251eafa58be91e2945da6','owner':'c75a3b5532700df673113ce6ae7b069042244569f4abf03c714ad0953c5098b7','harmonic_helper':'ed25e2f1a1549dab353ca4144c50d156e764a46e2aae504029c0ae853d4d5d7a','protection_source':'b6135f00fa83b366ee31f25e821a752a08edcfd926624a831d308cc7fd5ea2d3','boundary_report':'e753ec087610000dba23597e9f338cdbe94a0526809b290f02c8fb9f02824ce9','boundary_witnesses':'4d6fae8b49aac15634be8f6945e4d0bd8229d0b67f618a23d87207784bc0b2ea','facet_ancestry':'cbfebfc668cada5dc0a3e17f28f573904f805872cfdedd9dbf0980799bfbf5f4','role_map_report':'1fcb5f6df77acb65cc9ae21b470548496a0c793e25212c45260c626ab36e3e18','role_map_npz':'7c6da695e2d2e77f0404d828028cf828281f83b5daf1352d28ef04aa0a132ee1'}
MOVE_CAP=250e-6
RHO={15:900.0,16:850.0}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def npz(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def tri_multiset(faces):
 a=np.asarray(faces,np.int64).reshape(-1,3).copy();a.sort(axis=1);return a
def d3_exact(owner,points,face):
 a,b,c=[owner.f32_exact_point(points[int(i)]) for i in face]
 return (a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))
def exact_boundary_delta(owner,parent,candidate,moved_volume_rows):
 oldp=np.asarray(parent['points_world_m_f32'],np.float32);newp=np.asarray(candidate['points_world_m_f32'],np.float32);f=np.asarray(parent['boundary_faces'],np.int64)
 rows=np.asarray(sorted(moved_volume_rows),np.int64)
 hit=np.flatnonzero(np.isin(f,rows).any(axis=1)) if len(rows) else np.empty(0,np.int64)
 old=sum(d3_exact(owner,oldp,f[j]) for j in hit);new=sum(d3_exact(owner,newp,f[j]) for j in hit)
 return int(new-old),{'changed_boundary_faces':int(len(hit)),'old_boundary_numerator':str(old),'new_boundary_numerator':str(new),'delta_numerator':str(new-old)}
def row_csv(path,expected_sha,expected_count):
 if sha(path)!=expected_sha:raise ValueError('target CSV hash mismatch '+str(path))
 with Path(path).open(newline='') as f:
  rd=csv.DictReader(f)
  if rd.fieldnames!=['child_row']:raise ValueError('unexpected child-row schema')
  a=np.fromiter((int(r['child_row']) for r in rd),np.int64)
 if len(a)!=expected_count or len(np.unique(a))!=len(a):raise ValueError('target row count/uniqueness mismatch')
 return a
def main():
 print("root006 stage=input_and_topology_validation",flush=True)
 if OUT.exists():raise FileExistsError(OUT)
 for name,path in [('state',STATE),('typed',TYPED),('I',I_PATH),('O',O_PATH),('D',D_PATH),('Skin',SKIN_PATH),('Fat',FAT_PATH),('inventory',INV),('inventory_report',INV_REPORT),('census',CENSUS),('volume_bounds',VOLUME_BOUNDS),('target',TARGET_ALL),('owner',OWNER),('harmonic_helper',HELPER),('protection_source',PROTECTION_OWNER),('boundary_report',BOUNDARY_REPORT),('boundary_witnesses',BOUNDARY_WITNESSES),('facet_ancestry',FACET_ANCESTRY),('role_map_report',ROLE_MAP_REPORT),('role_map_npz',ROLE_MAP_NPZ)]:
  if not path.is_file():raise FileNotFoundError(path)
 for name,path in [('state',STATE),('typed',TYPED),('I',I_PATH),('O',O_PATH),('D',D_PATH),('Skin',SKIN_PATH),('Fat',FAT_PATH),('inventory',INV),('inventory_report',INV_REPORT),('census',CENSUS),('volume_bounds',VOLUME_BOUNDS),('target',TARGET_ALL),('owner',OWNER),('harmonic_helper',HELPER),('protection_source',PROTECTION_OWNER),('boundary_report',BOUNDARY_REPORT),('boundary_witnesses',BOUNDARY_WITNESSES),('facet_ancestry',FACET_ANCESTRY),('role_map_report',ROLE_MAP_REPORT),('role_map_npz',ROLE_MAP_NPZ)]:
  if sha(path)!=PINS[name]:raise ValueError('frozen input hash mismatch: '+name)
 owner=mod('point_update_owner_1009',OWNER);prep=mod('current1020_fit_source_007',PROTECTION_OWNER);harm=mod('current1020_harmonic_extension_001',HELPER)
 state_raw=json.loads(STATE.read_text());_,state=owner.unwrap_state(state_raw)
 if state.get('status')!='PASS_ZERO_SKIN_FAT_SOURCE_QUALITY_MASS_PENDING':raise ValueError('selected state status mismatch')
 assets=owner.current_asset_pins(state)
 for role,path in {'I':I_PATH,'O':O_PATH,'D':D_PATH,'Skin':SKIN_PATH,'Fat':FAT_PATH}.items():
  if Path(assets[role]['path'])!=path or assets[role]['sha256']!=PINS[role]:raise ValueError('selected state asset mismatch '+role)
 c=npz(TYPED,['points_reference_world_m_f64','volume_tetrahedra','material_index','geometric_donor_body','anatomical_region_id','current_typed_role_active_source_vertex_rows_I','current_typed_role_source_row_to_joint_node_I','current_typed_role_active_source_vertex_rows_O','current_typed_role_source_row_to_joint_node_O','current_typed_role_active_source_vertex_rows_D','current_typed_role_source_row_to_joint_node_D'])
 p64=np.asarray(c['points_reference_world_m_f64'],np.float64);p=p64.astype(np.float32).astype(np.float64);p32=p.astype(np.float32);t=np.asarray(c['volume_tetrahedra'],np.int64);mat=np.asarray(c['material_index'],np.int64)
 if p.shape!=(311768,3) or t.shape!=(1495294,4) or len(mat)!=len(t):raise ValueError('selected 433 mesh shape mismatch')
 surfaces={r:npz(path) for r,path in [('I',I_PATH),('O',O_PATH),('D',D_PATH)]};parents={'Skin':npz(SKIN_PATH),'Fat':npz(FAT_PATH)};source_maps=owner.check_source_maps(surfaces['I'],parents)
 map_report=json.loads(ROLE_MAP_REPORT.read_text());role_joint_maps=npz(ROLE_MAP_NPZ)
 if map_report.get('status')!='PASS_ALL_CELL_INCIDENCE_ROLE_JOINT_CROSSWALK' or map_report.get('physical_admission') is not False:raise ValueError('explicit role/joint map is not a source-only passing crosswalk')
 if map_report.get('maps',{}).get('sha256')!=PINS['role_map_npz']:raise ValueError('explicit role/joint map report does not bind NPZ')
 for role,path in {'Skin':SKIN_PATH,'Fat':FAT_PATH}.items():
  if map_report.get('role_inputs',{}).get(role,{}).get('sha256')!=PINS[role]:raise ValueError(role+' role/joint crosswalk parent input hash mismatch')
 role_maps={r:(np.asarray(c[f'current_typed_role_active_source_vertex_rows_{r}'],np.int64),np.asarray(c[f'current_typed_role_source_row_to_joint_node_{r}'],np.int64)) for r in ('O','I','D')}
 for role,(active,rowmap) in role_maps.items():
  faces=np.asarray(surfaces[role]['faces'],np.int64);pts=np.asarray(surfaces[role]['points_world_m_f32'],np.float32)
  if not np.array_equal(np.sort(active),np.unique(faces.reshape(-1))):raise ValueError(role+' active surface rows mismatch')
  if not np.array_equal(p32[rowmap[active]].view(np.uint32),pts[active].view(np.uint32)):raise ValueError(role+' exact F32 source/joint map mismatch')
 census=json.loads(CENSUS.read_text());bounds=json.loads(VOLUME_BOUNDS.read_text())
 if census.get('status')!='PASS_SOURCE_ONLY_COMPONENT_CENSUS' or census.get('component_count')!=9 or census.get('target_child_count')!=571297:raise ValueError('030c census qualification mismatch')
 if bounds.get('status')!='PASS_SOURCE_ONLY_REFERENCE_VOLUME_BOUNDS' or bounds.get('physical_admission') is not False or bounds.get('component_count')!=9:raise ValueError('030c volume bound report mismatch')
 if bounds.get('input_pins',{}).get('component_census',{}).get('sha256')!=PINS['census'] or bounds.get('input_pins',{}).get('typed_volume',{}).get('sha256')!=PINS['typed']:raise ValueError('030c volume report ancestry mismatch')
 comps=[];dens=[];goals=[];basevol=[];labels=[]
 for x,b in zip(census['components'],bounds['components']):
  cid=int(x['component_id']);rows=row_csv(x['target_rows_path'],x['target_rows_sha256'],int(x['target_child_count']))
  if int(b['component_id'])!=cid or int(b['target_child_count'])!=len(rows):raise ValueError('030c component census/bounds row mismatch')
  if rows.size and (rows.min()<0 or rows.max()>=len(t)):raise ValueError('030c component target row out of range')
  material_ids=np.unique(mat[rows])
  if len(material_ids)!=1 or int(material_ids[0])!=int(b['material_index']):raise ValueError('030c component material mismatch')
  density=float(b['minimum_density_kg_m3']);goal=float(b['minimum_volume_reduction_m3_for_isolated_component_bound'])
  if density<=0 or int(material_ids[0]) not in RHO or abs(RHO[int(material_ids[0])]-density)>1e-9:raise ValueError('030c density floor is not the pinned material rule')
  comps.append(rows);dens.append(density);goals.append(goal);basevol.append(float(b['current_reference_volume_m3']));labels.append(int(material_ids[0]))
 target_union=np.concatenate(comps);target_all=row_csv(TARGET_ALL,census['target_rows_sha256'],int(census['target_child_count']))
 if len(np.unique(target_union))!=len(target_union) or not np.array_equal(np.sort(target_union),np.sort(target_all)):raise ValueError('030c nine component rows do not partition authenticated target-side rows')
 # Exact source protections and root-supplied I roster.
 with np.load(INV,allow_pickle=False) as z:
  inv_i_rows=z['active_I_source_rows'].astype(np.int64);inv_i_nodes=z['active_I_joint_nodes'].astype(np.int64);inv_protected=z['protected_source_I_rows'].astype(np.int64)
 active_i,row_i=role_maps['I'];active_o,row_o=role_maps['O'];active_d,row_d=role_maps['D'];i_all=row_i[active_i];o_nodes=row_o[active_o];d_nodes=row_d[active_d]
 if not np.array_equal(inv_i_rows,active_i) or not np.array_equal(inv_i_nodes,i_all):raise ValueError('root I gradient inventory identity/order mismatch')
 back,routes,rims=prep.protection(owner,surfaces['I']);od_nodes=set(map(int,np.r_[o_nodes,d_nodes]));alias={int(r) for r in active_i if int(row_i[r]) in od_nodes};protected=set(back)|set(routes)|set(rims)|alias
 if not np.array_equal(np.asarray(sorted(protected),np.int64),inv_protected):raise ValueError('I protection roster differs from pinned root inventory before exterior-node protection')
 # Recompute exact tetra face incidence; every one-coface face is fixed, including the authored head-sidewall facets not present in O/D face tessellations.
 tet_faces=np.concatenate((t[:,[1,2,3]],t[:,[0,3,2]],t[:,[0,1,3]],t[:,[0,2,1]]),axis=0);canon_faces=tri_multiset(tet_faces);uniq,first,counts=np.unique(canon_faces,axis=0,return_index=True,return_counts=True)
 if np.any((counts!=1)&(counts!=2)):raise ValueError('current coupled tet face incidence is not manifold')
 boundary_faces=uniq[counts==1];boundary_owner=(first[counts==1]%len(t)).astype(np.int64);boundary_nodes=np.unique(boundary_faces.reshape(-1))
 ijfaces=tri_multiset(role_maps['I'][1][np.asarray(surfaces['I']['faces'],np.int64)])
 combined=np.concatenate((uniq,ijfaces),axis=0);uall,invall=np.unique(combined,axis=0,return_inverse=True);tet_count_by_face=np.zeros(len(uall),np.int8);tet_count_by_face[invall[:len(uniq)]]=counts.astype(np.int8);icount=tet_count_by_face[invall[len(uniq):]]
 if np.any(icount!=2):raise ValueError('current I surface is not exact internal two-coface interface')
 oid_nodes=np.unique(np.r_[i_all,o_nodes,d_nodes]);
 if not np.isin(boundary_nodes,oid_nodes).all():raise ValueError('current exterior contains nodes outside pinned O/I/D source-node union')
 odfaces=np.unique(tri_multiset(np.concatenate([role_maps[r][1][np.asarray(surfaces[r]['faces'],np.int64)] for r in ('O','D')],axis=0)),axis=0)
 joined=np.concatenate((boundary_faces,odfaces),axis=0);uj,ij=np.unique(joined,axis=0,return_inverse=True);nb=len(boundary_faces);bseen=np.zeros(len(uj),bool);oseen=np.zeros(len(uj),bool);bseen[ij[:nb]]=True;oseen[ij[nb:]]=True
 if np.count_nonzero(bseen&oseen)!=len(odfaces) or np.any(~bseen[ij[nb:]]):raise ValueError('not all exact O/D source facets occur on the current exterior')
 extra_mask=~oseen[ij[:nb]];extra=boundary_faces[extra_mask];extra_owner=boundary_owner[extra_mask]
 with np.load(BOUNDARY_WITNESSES,allow_pickle=False) as bw:
  if not np.array_equal(extra,bw['extra_tet_boundary_faces']) or not np.array_equal(extra_owner,bw['extra_tet_boundary_owner_rows']):raise ValueError('recomputed extra exterior facets disagree with pinned classifier witnesses')
 if len(extra)!=180 or not np.isin(extra,oid_nodes).all():raise ValueError('authored head-sidewall exterior classification changed')
 extra_labels,extra_counts=np.unique(np.stack((mat[extra_owner],np.asarray(c['geometric_donor_body'])[extra_owner],np.asarray(c['anatomical_region_id'])[extra_owner]),axis=1),axis=0,return_counts=True)
 wanted={(15,23,23):90,(16,23,23):90}
 if {tuple(map(int,r)):int(n) for r,n in zip(extra_labels,extra_counts)}!=wanted:raise ValueError('head-sidewall material/donor/region signature changed')
 # Bind each actual exterior extra facet to the authored 1044 headwall face ancestry, including material and child tet row.
 import csv
 extra_by_material={m:{tuple(sorted(map(int,face))):int(cell) for face,cell in zip(extra,extra_owner) if int(mat[cell])==m} for m in (15,16)}
 expected_roles={'donor_skin_OI_headwall':15,'donor_fat_ID_headwall':16}; matched={role:{} for role in expected_roles}
 with FACET_ANCESTRY.open(newline='') as fh:
  for row in csv.DictReader(fh):
   role=row['face_source_role']
   if role not in expected_roles:continue
   material=int(row['face_parent_material_index']);face=tuple(sorted(int(row[f'joint_node{k}'])-1 for k in range(3)))
   if material!=expected_roles[role] or face not in extra_by_material[material]:raise ValueError('authored headwall row does not map to exact exterior facet/material')
   if face in matched[role]:raise ValueError('duplicate authored headwall facet identity')
   if int(row['child_tet_row'])!=extra_by_material[material][face]:raise ValueError('headwall facet child-tet ancestry differs from actual boundary owner')
   matched[role][face]=row['child_tet_row']
 for role,material in expected_roles.items():
  if set(matched[role])!=set(extra_by_material[material]):raise ValueError('authored headwall facets do not exactly cover exterior sidewall material '+str(material))
 boundary_i=active_i[np.isin(row_i[active_i],boundary_nodes)]
 protected.update(map(int,boundary_i));prot=np.asarray(sorted(protected),np.int64)
 movable_source=np.asarray([int(r) for r in active_i if int(r) not in protected],np.int64);movable_nodes=row_i[movable_source]
 dirichlet=np.sort(np.asarray(role_joint_maps['O_I_D_boundary_joint_nodes'],np.int64))
 root_free=np.sort(np.asarray(role_joint_maps['free_joint_nodes'],np.int64))
 if len(np.unique(dirichlet))!=len(dirichlet) or len(np.unique(root_free))!=len(root_free):raise ValueError('explicit role/joint boundary/free rows are not unique')
 if not np.array_equal(np.sort(np.unique(np.r_[i_all,o_nodes,d_nodes])),dirichlet):raise ValueError('explicit role/joint boundary-node roster differs from active O/I/D source maps')
 if not np.array_equal(np.setdiff1d(np.arange(len(p),dtype=np.int64),dirichlet,assume_unique=True),root_free):raise ValueError('explicit free-joint rows are not the exact complement of O/I/D boundary rows')
 if not set(map(int,boundary_nodes)).issubset(set(map(int,dirichlet))):raise ValueError('actual exterior node not fixed in harmonic operator')
 if len(np.unique(movable_nodes))!=len(movable_nodes) or set(map(int,movable_nodes))&set(map(int,np.r_[o_nodes,d_nodes,boundary_nodes])):raise ValueError('movable I nodes alias another fixed boundary')
 qbase=owner.native_q(p32,t)
 if np.any(~np.isfinite(qbase)) or np.any(qbase<owner.Q_GATE):raise ValueError('selected 1020 baseline full-cell q gate is not clean')
 # Use the authenticated cell-incidence crosswalk; never infer role rows by global coordinate keys or legacy TetGen IDs.
 interior=root_free.copy()
 role_local_to_joint={'Skin':np.asarray(role_joint_maps['Skin_local_to_joint'],np.int64),'Fat':np.asarray(role_joint_maps['Fat_local_to_joint'],np.int64)}
 role_joint_to_local={};role_incident={};mapped_counts={};active_surface_volume_rows={};legacy_surface_alias_joint=set()
 for material_id,role in ((15,'Skin'),(16,'Fat')):
  local_to_joint=role_local_to_joint[role];points_role=np.asarray(parents[role]['points_world_m_f32'],np.float32);tets_role=np.asarray(parents[role]['tetrahedra'],np.int64)
  if local_to_joint.shape!=(len(points_role),):raise ValueError(role+' explicit node-map row count mismatch')
  active_local=np.flatnonzero(local_to_joint>=0)
  if np.any(local_to_joint[active_local]>=len(p)) or len(np.unique(local_to_joint[active_local]))!=len(active_local):raise ValueError(role+' active local-to-joint map is out of range or noninjective')
  if not np.array_equal(points_role[active_local].view(np.uint32),p32[local_to_joint[active_local]].view(np.uint32)):raise ValueError(role+' explicit role/local-to-joint point parity mismatch')
  if tets_role.size and (np.any(tets_role<0) or np.any(tets_role>=len(local_to_joint))):raise ValueError(role+' role tetrahedra contain out-of-range local nodes')
  mapped_role_tets=local_to_joint[tets_role]
  if np.any(mapped_role_tets<0):raise ValueError(role+' volume cell references an inactive -1 node map')
  typed_rows=np.flatnonzero(mat==material_id);typed_role_tets=t[typed_rows]
  if len(mapped_role_tets)!=len(typed_role_tets):raise ValueError(role+' role cell count differs from typed material domain')
  canonical_role=np.sort(mapped_role_tets,axis=1);canonical_typed=np.sort(typed_role_tets,axis=1)
  order_role=np.lexsort((canonical_role[:,3],canonical_role[:,2],canonical_role[:,1],canonical_role[:,0]));order_typed=np.lexsort((canonical_typed[:,3],canonical_typed[:,2],canonical_typed[:,1],canonical_typed[:,0]))
  if not np.array_equal(canonical_role[order_role],canonical_typed[order_typed]):raise ValueError(role+' role cell rows do not map exactly onto typed material cells')
  invmap=np.full(len(p),-1,np.int64);invmap[local_to_joint[active_local]]=active_local;role_joint_to_local[role]=invmap
  role_incident[role]=set(map(int,np.unique(typed_role_tets)))
  mapped_counts[role]={'allocated_local_rows':int(len(local_to_joint)),'active_local_rows':int(len(active_local)),'inactive_local_rows':int(np.count_nonzero(local_to_joint<0)),'mapped_cells':int(len(mapped_role_tets))}
  surf_used=set()
  for layer in (('O','I') if role=='Skin' else ('I','D')):
   source_active=np.unique(np.asarray(surfaces[layer]['faces'],np.int64).reshape(-1))
   map_key=f"{role.lower()}_current_target_rebind_{layer}_point_to_volume_rows"
   if map_key not in parents[role]:raise ValueError(role+' missing explicit current rebind map '+map_key)
   surface_map=np.asarray(parents[role][map_key],np.int64)
   if source_active.size and (source_active.max()>=len(surface_map) or np.any(surface_map[source_active]<0)):raise ValueError(role+' active '+layer+' source surface rows lack role-volume mappings')
   mapped_surface=surface_map[source_active]
   if not np.array_equal(local_to_joint[mapped_surface],role_maps[layer][1][source_active]):raise ValueError(role+' current '+layer+' surface map disagrees with authenticated typed joint rows')
   surf_used.update(map(int,mapped_surface))
  active_surface_volume_rows[role]=surf_used
  all_mapped_surface=set().union(*owner.volume_surface_point_rows(parents[role],role).values())
  legacy=all_mapped_surface-surf_used
  for j in interior:
   vr=int(invmap[int(j)])
   if vr<0:continue
   if vr in surf_used:raise ValueError(role+' authoritative free joint node is referenced by an active O/I/D surface face')
   if vr in legacy:legacy_surface_alias_joint.add(int(j))
 # A local point may have an obsolete map from an allocated but unreferenced surface row. Keep that allocation and map unchanged by fixing its exact volume node.
 frozen_legacy=np.asarray(sorted(legacy_surface_alias_joint),np.int64)
 interior=np.setdiff1d(root_free,frozen_legacy,assume_unique=True)
 dirichlet=np.unique(np.r_[dirichlet,frozen_legacy]).astype(np.int64)
 if not np.array_equal(np.setdiff1d(np.arange(len(p),dtype=np.int64),dirichlet,assume_unique=True),interior):raise ValueError('harmonic free roster differs after freezing legacy surface aliases')
 node_role_rows={int(j):{} for j in interior}
 for j in interior:
  expected_roles={r for r,s in role_incident.items() if int(j) in s}
  mapped_roles={r for r,invmap in role_joint_to_local.items() if int(invmap[int(j)])>=0}
  if expected_roles!=mapped_roles:raise ValueError('explicit per-role interior mapping does not match material-cell incidence')
  for role in expected_roles:node_role_rows[int(j)][role]=int(role_joint_to_local[role][int(j)])
 matched_counts={role:int(sum(role_joint_to_local[role][j]>=0 for j in interior)) for role in ('Skin','Fat')}
 ext=harm.HarmonicInteriorExtension(p,t,dirichlet,movable_nodes,material_ids=mat)
 if not np.array_equal(ext.interior_nodes,interior):raise ValueError('harmonic free rows differ from authenticated role/joint roster after legacy surface-row protection')
 # Verify and factor the existing harmonic push/pull implementation once.
 rng=np.random.default_rng(1020);probe_d=rng.normal(size=(len(movable_nodes),3))*1e-5;probe_g=rng.normal(size=(len(interior),3));probe_h=ext.push(probe_d);probe_ht=ext.pull(probe_g)
 lhs=float(np.sum(probe_h*probe_g,dtype=np.float64));rhs=float(np.sum(probe_d*probe_ht,dtype=np.float64));adj_error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-30)
 if not np.isfinite(probe_h).all() or not np.isfinite(probe_ht).all() or adj_error>1e-10:raise ValueError('harmonic push/pull adjoint identity failed')
 # Compute fresh 030c cut-volume derivatives; add exact harmonic-interior adjoint contribution.
 effective=[];volumes=[];cut_stats=[]
 for cid,rows in enumerate(comps):
  g,v,det=prep.grad_for(p64,t,rows,len(p))
  if np.any(det<=0):raise ValueError('030c target contains nonpositive F64 reference tetrahedra')
  if abs(v-basevol[cid])>max(1e-13,basevol[cid]*1e-9):raise ValueError('030c volume bounds do not match current typed coordinates')
  gi=g[movable_nodes]+ext.pull(g[interior])
  effective.append(gi);volumes.append(v)
  cut_stats.append({'component_id':cid,'material_index':labels[cid],'target_cell_count':len(rows),'volume_m3':v,'required_reduction_m3':goals[cid],'minimum_density_kg_m3':dens[cid],'effective_gradient_nonzero_I_nodes':int(np.count_nonzero(np.linalg.norm(gi,axis=1)>0)),'effective_gradient_l1_m2':float(np.linalg.norm(gi,axis=1).sum())})
 effective=np.asarray(effective,np.float64)
 OUT.mkdir(parents=True)
 # Make one coherent low-frequency boundary field. Positive LP coefficients combine descent bases while avoiding a first-order increase in any target component.
 faces=np.asarray(surfaces['I']['faces'],np.int64);blocked=set(protected);trial_log=[];accepted=None
 def build_direction(blocked_rows):
  bases=[]
  for k in range(len(comps)):
   f=np.zeros((len(surfaces['I']['points_world_m_f32']),3),np.float64);f[movable_source]=-effective[k]
   bases.append(prep.smooth(f,faces,blocked_rows,2)[movable_source])
  bases=np.asarray(bases);a=np.asarray([[np.sum(effective[i]*bases[j]) for j in range(len(comps))] for i in range(len(comps))],np.float64)
  weights=1.0/np.maximum(np.asarray(goals),1e-12)
  # a is signed dV, so minimizing weighted a selects volume reduction.
  # Scale rows to avoid HiGHS feasibility tolerances swallowing m^4 entries.
  row_scale=np.maximum(np.max(np.abs(a),axis=1),np.finfo(np.float64).tiny)
  objective=weights@a
  objective/=max(float(np.max(np.abs(objective))),np.finfo(np.float64).tiny)
  res=linprog(objective,A_ub=a/row_scale[:,None],b_ub=np.zeros(len(comps)),A_eq=np.ones((1,len(comps))),b_eq=np.ones(1),bounds=[(0,None)]*len(comps),method='highs')
  if not res.success:raise ValueError('no nonnegative coherent no-increase descent combination: '+res.message)
  direction=np.einsum('k,kij->ij',res.x,bases)
  di=np.zeros((len(surfaces['I']['points_world_m_f32']),3),np.float64);di[movable_source]=direction
  interior_d=ext.push(direction)
  request=max(float(np.linalg.norm(direction,axis=1).max(initial=0.)),float(np.linalg.norm(interior_d,axis=1).max(initial=0.)))
  if not np.isfinite(request) or request==0:raise ValueError('zero/nonfinite harmonic fit direction')
  scale=min(1.0,MOVE_CAP/request)
  return direction*scale,interior_d*scale,a,res.x,request,scale
 for pass_no in range(4):
  print('root006 active_set_pass='+str(pass_no)+' blocked_I_rows='+str(len(blocked)),flush=True)
  direction,interior_direction,a,coeff,requested,scale=build_direction(blocked)
  oldp_i=np.asarray(surfaces['I']['points_world_m_f32'],np.float32);old_joint=p32.copy();failing_i=set();pass_chosen=None
  for alpha in (1.0,0.5,0.25,0.125,0.0625,0.03125):
   print('root006 trial_alpha='+str(alpha),flush=True)
   new_joint=old_joint.copy();new_joint[movable_nodes]=(p[movable_nodes]+alpha*direction).astype(np.float32);new_joint[interior]=(p[interior]+alpha*interior_direction).astype(np.float32)
   new_i=oldp_i.copy();new_i[active_i]=new_joint[row_i[active_i]]
   changed_i=np.flatnonzero(np.any(new_i.view(np.uint32)!=oldp_i.view(np.uint32),axis=1))
   changed_joint=np.flatnonzero(np.any(new_joint.view(np.uint32)!=p32.view(np.uint32),axis=1))
   if not len(changed_joint):trial_log.append({'active_set_pass':pass_no,'alpha':alpha,'status':'NO_F32_CHANGE'});continue
   if set(map(int,changed_i))&blocked:raise ValueError('active-set protected I source row moved')
   fchecks,fsum=prep.face_checks(oldp_i,new_i,faces,set(map(int,changed_i)))
   if not fsum['pass']:
    failfaces=[int(x['row']) for x in fchecks if x['orientation_cosine']<.5 or x['area_ratio']<.5 or x['area_ratio']>2.0]
    if failfaces:
     failing_i.update(map(int,np.unique(faces[np.asarray(failfaces,np.int64)].reshape(-1))))
    trial_log.append({'active_set_pass':pass_no,'alpha':alpha,'status':'I_FACE_SHAPE_REJECT','face_summary':fsum,'failing_faces':failfaces[:16]});continue
   qnew=owner.native_q(new_joint,t);badq=np.flatnonzero((~np.isfinite(qnew))|(qnew<owner.Q_GATE));old_exact=[];new_exact=[]
   stars=owner.star_cells(t,set(map(int,changed_joint)))
   if not len(badq):
    for cell in stars:
     if owner.exact_det6(p32,t[int(cell)])<=0:old_exact.append(int(cell))
     if owner.exact_det6(new_joint,t[int(cell)])<=0:new_exact.append(int(cell))
   print('root006 quality_bad_q='+str(len(badq))+' exact_nonpositive='+str(len(new_exact)),flush=True)
   if len(badq) or old_exact or new_exact:
    badcells=set(map(int,badq))|set(old_exact)|set(new_exact)
    if badcells:
     badnodes=np.unique(t[np.asarray(sorted(badcells),np.int64)].reshape(-1));expand=owner.star_cells(t,set(map(int,badnodes)));expandnodes=np.unique(t[expand].reshape(-1))
     failing_i.update(map(int,active_i[np.isin(row_i[active_i],expandnodes)]))
    trial_log.append({'active_set_pass':pass_no,'alpha':alpha,'status':'TET_QUALITY_REJECT','bad_q_count':int(len(badq)),'bad_q_rows':badq[:12].tolist(),'old_exact_nonpositive_rows':old_exact[:12],'new_exact_nonpositive_rows':new_exact[:12],'changed_joint_nodes':int(len(changed_joint)),'expanded_source_I_freeze_rows':len(failing_i)});continue
   component_after=[prep.volume_sum(new_joint,t,r) for r in comps];reductions=np.asarray(volumes)-np.asarray(component_after)
   if np.any(reductions < -1e-12) or not np.any(reductions>1e-12):
    trial_log.append({'active_set_pass':pass_no,'alpha':alpha,'status':'COMPONENT_VOLUME_DIRECTION_REJECT','component_reduction_m3':reductions.tolist()});continue
   src_updates={int(r):{'old':oldp_i[int(r)].copy(),'new':new_i[int(r)].copy()} for r in changed_i}
   int_updates=[]
   for j in interior:
    if np.array_equal(new_joint[int(j)].view(np.uint32),p32[int(j)].view(np.uint32)):continue
    for role,vr in node_role_rows[int(j)].items():int_updates.append({'role':role,'row':int(vr),'old':np.asarray(parents[role]['points_world_m_f32'][vr],np.float32).copy(),'new':new_joint[int(j)].copy()})
   candidate_dir=OUT/'candidate-step-001';paths={'I':candidate_dir/'candidate-I-current1020-030c.npz','Skin':candidate_dir/'skin-current1020-030c.npz','Fat':candidate_dir/'fat-current1020-030c.npz'}
   iout,vout,plan=owner.apply_plan(surfaces['I'],parents,src_updates,int_updates,assets,paths)
   expected_state=dict(state);expected_state['_candidate_report_source_face_checks']={'rows':fchecks}
   previous_global_cap=float(owner.GLOBAL_VOLUME_CAP);owner.GLOBAL_VOLUME_CAP=1.0
   pair_error=None;pair=None
   try:pair=owner.replay_pair(parents,vout,{'I':iout,**vout},expected_state,surfaces['I'],iout,src_updates,int_updates,source_maps,0.0)
   except Exception as exc:pair_error=type(exc).__name__+': '+str(exc)
   finally:owner.GLOBAL_VOLUME_CAP=previous_global_cap
   boundary_checks={};exact_pair_ok=False
   if pair is not None:
    face_deltas=[];tet_deltas=[]
    for role in ('Skin','Fat'):
     moved_rows=set(map(int,source_maps[role][changed_i]));moved_rows.update(int(u['row']) for u in int_updates if u['role']==role)
     bd,bdinfo=exact_boundary_delta(owner,parents[role],vout[role],moved_rows);td=int(pair['roles'][role]['changed_star_exact_det6_delta_integer'])
     boundary_checks[role]={**bdinfo,'exact_changed_tet_det6_delta_integer':str(td),'exact_boundary_delta_equals_tet_delta':bd==td,'global_volume_delta_fraction':pair['roles'][role]['global_exact_volume_delta_fraction'],'local_volume_delta_fraction':pair['roles'][role]['local_exact_volume_delta_fraction']}
     face_deltas.append(bd);tet_deltas.append(td)
    exact_pair_ok=(all(x['exact_boundary_delta_equals_tet_delta'] for x in boundary_checks.values()) and sum(tet_deltas)==0 and sum(face_deltas)==0)
   pair_ok=(pair_error is None and exact_pair_ok)
   result={'active_set_pass':pass_no,'alpha':alpha,'status':'PASS_COUPLED_SOURCE_QUALITY' if pair_ok else 'ROLE_REPLAY_OR_EXACT_BOUNDARY_REJECT','changed_I_source_rows':int(len(changed_i)),'changed_joint_nodes':int(len(changed_joint)),'changed_interior_joint_nodes':int(np.count_nonzero(np.any(new_joint[interior].view(np.uint32)!=p32[interior].view(np.uint32),axis=1))),'changed_433_star_cells':int(len(stars)),'candidate_min_q':float(np.min(qnew)),'candidate_bad_q':int(len(badq)),'changed_star_old_exact_nonpositive':old_exact[:12],'changed_star_new_exact_nonpositive':new_exact[:12],'source_I_face_summary':fsum,'component_reference_volumes_m3':volumes,'component_candidate_volumes_m3':component_after,'component_actual_reductions_m3':reductions.tolist(),'component_first_order_reductions_m3':(-alpha*(a@coeff*scale)).tolist(),'first_order_matrix_m2':a.tolist(),'descent_coefficients':coeff.tolist(),'maximum_I_or_interior_motion_m':float(max(np.linalg.norm((new_i[changed_i].astype(np.float64)-oldp_i[changed_i]),axis=1).max(initial=0.),np.linalg.norm(new_joint[interior].astype(np.float64)-p[interior],axis=1).max(initial=0.))),'paired_role_error':pair_error,'paired_replay':pair,'exact_boundary_volume_checks':boundary_checks,'exact_boundary_and_pair_conservation_pass':exact_pair_ok,'point_update_plan':plan}
   trial_log.append(result)
   if pair_ok:
    pass_chosen=(new_joint,new_i,changed_i,changed_joint,stars,src_updates,int_updates,fchecks,fsum,pair,plan,boundary_checks,reductions,coeff,requested,scale,alpha)
    break
  if pass_chosen:
   accepted=pass_chosen;break
  if not failing_i:break
  newly=failing_i-set(blocked)
  if not newly:break
  blocked.update(newly)
 if accepted is None:
  status='NO_ACCEPTED_BOUNDED_COUPLED_STEP';final_assets={};final_joint=None
 else:
  new_joint,new_i,changed_i,changed_joint,stars,src_updates,int_updates,fchecks,fsum,pair,plan,boundary_checks,reductions,coeff,requested,scale,alpha=accepted
  candidate_dir=OUT/'candidate-step-001';candidate_dir.mkdir(parents=True)
  paths={'I':candidate_dir/'candidate-I-current1020-030c.npz','Skin':candidate_dir/'skin-current1020-030c.npz','Fat':candidate_dir/'fat-current1020-030c.npz'}
  iout,vout,plan=owner.apply_plan(surfaces['I'],parents,src_updates,int_updates,assets,paths)
  stage=Path(tempfile.mkdtemp(prefix='.stage-',dir=str(candidate_dir)));ire=owner.write_npz_and_verify(stage/paths['I'].name,iout)
  for role in ('Skin','Fat'):
   if 'typed_I_candidate_sha256' in vout[role]:owner.set_string_like(vout[role],'typed_I_candidate_sha256',ire['sha256'])
  vrecs={role:owner.write_npz_and_verify(stage/paths[role].name,vout[role]) for role in ('Skin','Fat')}
  reread_i=owner.load_npz(stage/paths['I'].name);reread_v={r:owner.load_npz(stage/paths[r].name) for r in ('Skin','Fat')}
  expected_state=dict(state);expected_state['_candidate_report_source_face_checks']={'rows':fchecks}
  previous_global_cap=float(owner.GLOBAL_VOLUME_CAP);owner.GLOBAL_VOLUME_CAP=1.0
  try:pair2=owner.replay_pair(parents,reread_v,{'I':reread_i,**reread_v},expected_state,surfaces['I'],reread_i,src_updates,int_updates,source_maps,0.0)
  finally:owner.GLOBAL_VOLUME_CAP=previous_global_cap
  if pair2['paired_exact_volume_cancel'] is not True:raise ValueError('serialized coupled role replay lost exact pair cancellation')
  reread_bd=[];reread_td=[]
  for role in ('Skin','Fat'):
   moved_rows=set(map(int,source_maps[role][changed_i]));moved_rows.update(int(u['row']) for u in int_updates if u['role']==role)
   bd,_=exact_boundary_delta(owner,parents[role],reread_v[role],moved_rows);td=int(pair2['roles'][role]['changed_star_exact_det6_delta_integer'])
   if bd!=td:raise ValueError('serialized exact boundary/tet volume identity mismatch for '+role)
   reread_bd.append(bd);reread_td.append(td)
  if sum(reread_bd)!=0 or sum(reread_td)!=0:raise ValueError('serialized exact paired boundary volume does not cancel')
  for role in ('I','Skin','Fat'):shutil.move(str(stage/paths[role].name),str(candidate_dir/paths[role].name))
  stage.rmdir();final_assets={r:{'path':str(candidate_dir/paths[r].name),'sha256':sha(candidate_dir/paths[r].name),'bytes':candidate_dir.joinpath(paths[r].name).stat().st_size} for r in ('I','Skin','Fat')};final_joint=new_joint;status='PASS_COUPLED_030C_SOURCE_CANDIDATE_PENDING_SURFACE_AND_MASS_QUALIFICATION'
 roster=OUT/'harmonic-node-roster.npz'
 node_role_rows_arr=np.asarray([[int(j),node_role_rows[int(j)].get('Skin',-1),node_role_rows[int(j)].get('Fat',-1)] for j in interior],np.int64)
 np.savez_compressed(roster,dirichlet_joint_nodes=dirichlet,movable_i_source_rows=movable_source,movable_i_joint_nodes=movable_nodes,protected_i_source_rows=prot,fixed_o_source_rows=active_o,fixed_o_joint_nodes=o_nodes,fixed_d_source_rows=active_d,fixed_d_joint_nodes=d_nodes,interior_joint_nodes=interior,interior_role_rows=node_role_rows_arr,i_face_joint_rows=role_maps['I'][1][np.asarray(surfaces['I']['faces'],np.int64)])
 if accepted is not None:
  new_joint=final_joint;mov=np.flatnonzero(np.any(new_joint.view(np.uint32)!=p32.view(np.uint32),axis=1));motion=OUT/'candidate-coupled-motion.npz';np.savez_compressed(motion,changed_joint_nodes=mov,changed_points_world_m_f32=new_joint[mov],active_i_source_rows=changed_i,active_i_points_world_m_f32=new_i[changed_i],interior_joint_nodes=interior,interior_displacement_m=(new_joint[interior].astype(np.float64)-p[interior]),harmonic_helper_sha256=np.asarray([sha(HELPER)]))
 else:motion=None
 report={'schema':'numi.h706.current1020-coupled-harmonic-030c-I-redistribution-candidate.v1','status':status,'physical_admission':False,'geometry_source':'source-only current 1020/433 selected reference; candidate is separate and not selected','native_steps':0,'mass_mapping':'NOT_RUN','inputs':{k:pin(v) for k,v in {'state':STATE,'typed433':TYPED,'I':I_PATH,'O':O_PATH,'D':D_PATH,'Skin':SKIN_PATH,'Fat':FAT_PATH,'root_sensitivities':INV,'root_inventory_report':INV_REPORT,'030c_component_census':CENSUS,'030c_volume_bounds':VOLUME_BOUNDS,'030c_target_rows':TARGET_ALL,'point_update_owner':OWNER,'harmonic_helper':HELPER,'protection_source':PROTECTION_OWNER,'boundary_classification':BOUNDARY_REPORT,'boundary_witnesses':BOUNDARY_WITNESSES,'facet_ancestry':FACET_ANCESTRY,'cell_incidence_role_joint_map_report':ROLE_MAP_REPORT,'cell_incidence_role_joint_map_npz':ROLE_MAP_NPZ}.items()},'protected_scope':{'O_fixed_source_rows':len(active_o),'D_fixed_source_rows':len(active_d),'I_active_source_rows':len(active_i),'I_protected_source_rows':len(prot),'I_actual_exterior_source_rows':len(boundary_i),'I_movable_source_rows':len(movable_source),'I_surface_rims':len(rims),'back_rows':len(back),'route_rows':len(routes),'O_D_alias_I_rows':len(alias)},'harmonic_operator':{'schema_source':pin(HELPER),'P1_coupled_material_mesh':True,'joint_nodes':len(p),'tets':len(t),'Dirichlet_joint_nodes':len(dirichlet),'true_interior_joint_nodes':len(interior),'authenticated_root_free_joint_nodes':len(root_free),'frozen_legacy_surface_alias_joint_nodes':len(frozen_legacy),'movable_I_joint_nodes':len(movable_nodes),'operator_factorization_count':ext.audit.factorization_count,'A_ii_nnz':int(ext.A_ii.nnz),'A_iI_nnz':int(ext.A_iI.nnz),'push_pull_adjoint_relative_error':adj_error,'interior_exact_role_local_rows':matched_counts,'explicit_cell_incidence_role_joint_map':pin(ROLE_MAP_NPZ),'explicit_crosswalk_report':pin(ROLE_MAP_REPORT),'role_local_cell_and_point_parity':mapped_counts,'root_free_joint_nodes':int(len(root_free)),'frozen_unreferenced_surface_alias_joint_nodes':int(len(frozen_legacy)),'interior_material_role_map':'authoritative cell-incidence crosswalk with per-cell 4-corner identity, all incident-cell vertex consistency, and exact F32 parity; no global point-key or legacy TetGen-ID inference'},'boundary_topology':{'tet_exterior_faces':int(len(boundary_faces)),'exact_O_D_source_facets_are_on_exterior':True,'authored_head_sidewall_extra_facets':int(len(extra)),'authored_headwall_ancestry_csv':pin(FACET_ANCESTRY),'headwall_exact_matches_by_role':{role:len(rows) for role,rows in matched.items()},'extra_sidewall_label_counts':{'15:23:23':90,'16:23:23':90},'actual_exterior_nodes_fixed':int(len(boundary_nodes)),'active_I_rows_frozen_on_actual_exterior':int(len(boundary_i)),'I_interface_faces':int(len(ijfaces)),'all_I_faces_have_two_tet_cofaces':True,'tet_face_incidence_only_one_or_two':True},'current030c_targets':{'schema':census['schema'],'component_count':len(comps),'target_child_count':len(target_all),'target_rows_sha256':sha(TARGET_ALL),'volume_bounds_report':pin(VOLUME_BOUNDS),'interpretation':bounds['interpretation'],'components':cut_stats},'fit':{'max_requested_before_250um_cap_m':accepted[14] if accepted else None,'requested_displacement_scale':accepted[15] if accepted else None,'accepted_alpha':accepted[16] if accepted else None,'move_cap_m':MOVE_CAP,'blocked_I_source_rows_after_active_set':len(blocked-protected),'direction_objective':'nonnegative combination of smoothed effective gradients, constrained to first-order nonincrease of every 030c target component; effective gradient includes H^T times all true-interior volume gradient','active_set_passes':len({x['active_set_pass'] for x in trial_log}),'trial_log':trial_log,'component_first_order_matrix_m2':next((x['first_order_matrix_m2'] for x in trial_log if x['status']=='PASS_COUPLED_SOURCE_QUALITY'),None),'accepted_assets':final_assets,'serialized_candidate_motion':pin(motion) if motion else None,'harmonic_node_roster':pin(roster),'paired_role_volume_policy':{'historical_global_volume_cap_1e-4_applied':False,'reason':'replaced for this I-redistribution candidate by exact per-role boundary-volume delta == exact changed-tet det6 delta, plus exact Skin+Fat total volume cancellation; original owner file remains unchanged','replay_owner_local_volume_cap_0.01_retained':True,'O_D_fixed':True,'paired_exact_cancellation':bool(accepted is not None),'boundary_delta_checks':accepted[11] if accepted else {}},'surface_self_cross_and_806_808':'PENDING_NOT_RUN'},'limitations':'Current 030c isolated component volume reductions are necessary local construction bounds only, not proof of simultaneous flow feasibility or an accepted anatomy change. This candidate has no fresh 806/808, source-surface self/cross audit, mass mapping, native or physical admission.'}
 rp=OUT/'report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':status,'report':pin(rp),'roster':pin(roster),'motion':pin(motion) if motion else None,'accepted_assets':final_assets,'operator':report['harmonic_operator'],'trial_count':len(trial_log)},indent=2))
if __name__=='__main__':main()
