"""Independent exact replay of construction-only LP interior relocations."""
from pathlib import Path
import json,hashlib,importlib.util,argparse,numpy as np,time,csv
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=mod('volume456',H/'transfer_surface_contractions_indexed_379_root.py');R=mod('registration456',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
def registered_fat_source_rows(registration_map, source_points):
    """Return source-row identity mask from explicit registered role records.

    Historical source-node coordinate matches alone do not confer a registered
    surface identity. O/I rows must carry direct current typed-vertex identity;
    volume-interior rows must explicitly declare no surface ancestry.
    """
    n = len(source_points)
    seen = np.zeros(n, dtype=bool)
    registered = np.zeros(n, dtype=bool)
    counts = {"D": 0, "I": 0, "interior_no_surface_ancestry": 0}
    vertex_rows = {"D": set(), "I": set()}
    surface_vertex_rows = {"D": set(), "I": set()}
    with Path(registration_map).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_partition", "source_tetgen_node_id", "role",
                    "source_domain", "source_mapping_kind", "source_vertex_row",
                    "source_face_row", "registered_surface_layer",
                    "registered_surface_vertex_row", "current_x_m", "current_y_m",
                    "current_z_m", "parent_registration_kind", "parent_tet_row",
                    "anatomical_region_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("registered node map lacks required identity columns")
        for x in reader:
            if x["source_partition"] != "fat":
                continue
            sid = int(x["source_tetgen_node_id"])
            row = sid - 1
            if row < 0 or row >= n or seen[row]:
                raise ValueError("fat TetGen source node IDs are duplicate or out of range")
            seen[row] = True
            mapped = np.asarray([float(x["current_x_m"]), float(x["current_y_m"]),
                                 float(x["current_z_m"])], dtype=np.float32)
            if not np.array_equal(mapped.view(np.uint32), source_points[row].view(np.uint32)):
                raise ValueError("registration-map coordinates disagree with pinned source node row")
            role = x["role"]
            domain = x["source_domain"]
            kind = x["source_mapping_kind"]
            source_vertex = int(x["source_vertex_row"])
            source_face = int(x["source_face_row"])
            surface_layer = x["registered_surface_layer"]
            surface_vertex = int(x["registered_surface_vertex_row"])
            if role in ("D", "I"):
                if domain != "derived_registered_skin_authoring" or kind != "vertex":
                    raise ValueError("O/I point lacks direct registered-surface vertex identity")
                if source_vertex < 0 or source_face != -1 or surface_layer != role or surface_vertex < 0:
                    raise ValueError("O/I point has incomplete or conflicting registered identity")
                if source_vertex in vertex_rows[role] or surface_vertex in surface_vertex_rows[role]:
                    raise ValueError("registered O/I vertex identity is not unique")
                vertex_rows[role].add(source_vertex)
                surface_vertex_rows[role].add(surface_vertex)
                registered[row] = True
                counts[role] += 1
            elif role == "interior":
                if domain != "derived_volume_interior_no_surface_ancestry" or kind != "none":
                    raise ValueError("interior row is not explicitly marked no-surface-ancestry")
                if source_vertex != -1 or source_face != -1 or surface_layer or surface_vertex != -1:
                    raise ValueError("inferred volume interior carries unexpected surface identity")
                if int(x["parent_tet_row"]) < 0 or int(x["anatomical_region_id"]) < 0:
                    raise ValueError("inferred volume interior lacks current inferred parent registration")
                if x["parent_registration_kind"] != "inferred_current_parent_tet_same_material_donor_region":
                    raise ValueError("inferred volume interior parent-registration method is unexpected")
                counts["interior_no_surface_ancestry"] += 1
            else:
                raise ValueError("unknown Fat source-node role in registration map")
    if not np.all(seen):
        raise ValueError(f"registration map omits {int(np.sum(~seen))} Fat source-node rows")
    return registered, counts

def fixed_current_registered_points(p):
 report=H/'fat-inferred-identity-mask-336-root/report.json'
 r=json.loads(report.read_text())
 assert r['status']=='PASS_FAT_REGISTERED_IDENTITY_AND_PENDING_CAVITY_MASK'
 assert r['source_input_sha256']=='4602edc3b1db3617ab610c21f91309db10892c0e28ff7d6ba5c381491d13df2c'
 assert sha(r['source_input'])==r['source_input_sha256']
 mask=H/'fat-inferred-identity-mask-336-root/identity-and-pending-cavity-mask.npz'
 assert sha(mask)==r['mask_sha256']=='b80ced216b95f0d7bdb1245252e11b6742f93dda31793dfe121b1204f61058f5'
 assert sha(r['registered_map'])==r['registered_map_sha256']
 with np.load(r['source_input']) as z:oldp=z['points_world_m']
 with np.load(mask) as z:protected=z['protected_point_rows'];assert str(z['source_input_sha256'].item())==r['source_input_sha256']
 lookup={T.key(x):i for i,x in enumerate(p)};assert len(lookup)==len(p)
 # Existing336 explicit semantic identity ownership; never infer registration
 # merely because a point matches an arbitrary historical interior.
 current=np.asarray([lookup[T.key(oldp[i])] for i in protected],np.int64)
 assert len(current)==len(np.unique(current))==r['protected_existing_point_rows']
 assert np.array_equal(p[current].view(np.uint32),oldp[protected].view(np.uint32))
 fixed=np.zeros(len(p),bool);fixed[current]=True
 return fixed,{'existing336_report':str(report),'existing336_report_sha256':sha(report),'existing336_mask':str(mask),'existing336_mask_sha256':sha(mask),'exact_registered_points_mapped':len(current),'identity_counts':r['identity_counts'],'mapping':'exactF32coordinatebijectionofauthenticated336registeredrows'}

a=argparse.ArgumentParser();a.add_argument('--report',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();args.output.mkdir(exist_ok=False);start=time.monotonic()
r=json.loads(args.report.read_text());ip=Path(r['input']);op=Path(r['candidate']);assert sha(ip)==r['input_sha256'] and sha(op)==r['candidate_sha256']
with np.load(ip) as z:old={k:z[k] for k in z.files}
with np.load(op) as z:new={k:z[k] for k in z.files}
for key in old:
 if key not in ('points_world_m','points_world_m_f32'):assert np.array_equal(old[key],new[key]),key
p=old['points_world_m'].copy();t=old['tetrahedra'];mat=old['material_indices'];d,q=T.m.measurements(p,t);good=T.m.passes(d,q)
boundary=T.boundary(t);fixed=np.zeros(len(p),bool);fixed[np.unique(boundary)]=True
assert sha(r['registration_map'])==r['registration_map_sha256']
registered,identity_proof=fixed_current_registered_points(p);fixed|=registered
assert identity_proof==r['registered_identity_crosswalk']
for protection in r['protected_cell_maps']:
 assert sha(protection['path'])==protection['sha256'] and sha(protection['reference_volume'])==protection['reference_sha256']
 with np.load(protection['reference_volume']) as z:assert np.array_equal(z['points_world_m'].view(np.uint32),p.view(np.uint32));reference_t=z['tetrahedra']
 with np.load(protection['path']) as z:
  if 'comp0_full_protected_cell_rows' in z:rows=z['comp0_full_protected_cell_rows'];pts=z['comp0_full_protected_point_rows']
  else:rows=z['input_cell_rows'];pts=z['global_point_rows']
 footprint=np.unique(np.r_[pts,reference_t[rows].ravel()]);assert len(footprint)==protection['protected_corner_points'];fixed[footprint]=True

used=set();signs=0;exact_delta=0;maxd=0.
for entry in r['operations']:
 node=int(entry['node']);assert not fixed[node]
 rows=np.asarray(entry['cell_rows'],np.int64);assert np.array_equal(rows,np.flatnonzero(np.any(t==node,axis=1)))
 assert not used&set(map(int,rows));used.update(map(int,rows))
 assert len(np.unique(mat[rows]))==1 and np.array_equal(p[node].view(np.uint32),np.asarray(entry['old_f32'],np.float32).view(np.uint32))
 oldsum=sum(R.exact_star_determinants(p,t[rows]));assert str(oldsum)==entry['exact_det6_sum']
 oldpass=T.m.passes(*T.m.measurements(p,t[rows]));before=p[node].copy();p[node]=np.asarray(entry['new_f32'],np.float32)
 nd,nq=T.m.measurements(p,t[rows]);newpass=T.m.passes(nd,nq)
 assert np.all(nd>0) and not np.any(oldpass&~newpass)
 assert int((~oldpass).sum())==entry['old_bad'] and int((~newpass).sum())==entry['new_bad'] and entry['new_bad']<entry['old_bad']
 ex=R.exact_star_determinants(p,t[rows]);assert min(ex)>0 and sum(ex)==oldsum;signs+=len(ex)
 displacement=float(np.linalg.norm(p[node].astype(float)-before.astype(float)));assert displacement==entry['displacement_m'] and displacement<=entry['maximum_displacement_m']
 maxd=max(maxd,displacement)
assert np.array_equal(p.view(np.uint32),new['points_world_m'].view(np.uint32))
assert np.array_equal(p.view(np.uint32),new['points_world_m_f32'].view(np.uint32))
assert np.array_equal(p[fixed].view(np.uint32),old['points_world_m'][fixed].view(np.uint32))
assert len(np.unique(p,axis=0))==len(np.unique(old['points_world_m'],axis=0))
nd,nq=T.m.measurements(p,t);ng=T.m.passes(nd,nq)
assert np.all(nd>0) and not np.any(good&~ng) and int((~ng).sum())==r['bad_after']
assert np.array_equal(T.boundary_identity(boundary),T.boundary_identity(T.boundary(new['tetrahedra'])))
assert np.array_equal(old['desired_boundary_faces'],new['desired_boundary_faces'])
receipt={'status':'PASS_EXACT_FIXED_BOUNDARY_INTERIOR_LP_REPLAY','physical_admission':False,'native_steps':0,'input':str(ip),'input_sha256':sha(ip),'candidate':str(op),'candidate_sha256':sha(op),'construction_report':str(args.report),'construction_report_sha256':sha(args.report),'bad_before':r['bad_before'],'bad_after':r['bad_after'],'operations':len(r['operations']),'exact_positive_signs_checked':signs,'exact_local_and_global_volume_delta':0,'actual_boundary_source_geometry_material_connectivity_preserved':True,'complete_target_boundary':'NOT_NEWLY_QUALIFIED_INHERITED_RESIDUAL_UNCHANGED','protected_cell_maps':r['protected_cell_maps'],'all_registered_points_bit_identical':True,'all_point_coordinate_views_synchronized':True,'maximum_inferred_interior_displacement_m':maxd,'source330_mass':'IMMUTABLE_CURRENT_MAPPING_REQUIRED','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
