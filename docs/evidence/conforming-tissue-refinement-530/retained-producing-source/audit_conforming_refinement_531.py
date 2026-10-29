from pathlib import Path
import numpy as np,json,sys,importlib.util,hashlib,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
out=P/'conforming-refinement-control-530/run-001/exact-topology-531';out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(P/'conforming-refinement-control-530/run-001/refinement-connectivity.npz') as z:r={k:z[k].copy() for k in z.files}
with np.load(P/'reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz') as z:a={k:z[k].copy() for k in z.files}
x=r['points_reference_f64'].astype(np.float32);t=r['tetrahedra'];cp=r['child_source_tet_rows'];mat=a['material_index'][cp]
ordered=t[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3)
keys=np.sort(ordered,axis=1);u,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
parity=((ordered[:,0]>ordered[:,1]).astype(int)+(ordered[:,0]>ordered[:,2]).astype(int)+(ordered[:,1]>ordered[:,2]).astype(int))%2
balance=np.bincount(inv,weights=1-2*parity,minlength=len(u));owners=np.repeat(mat,4)
is15=np.bincount(inv,weights=(owners==15),minlength=len(u));is16=np.bincount(inv,weights=(owners==16),minlength=len(u))
shared=(is15>0)&(is16>0)
role=np.repeat(np.arange(3),len(a['points_reference_world_m_f64'])//3).tolist()
for aa,bb in r['inserted_edge_endpoints']:
 assert role[int(aa)]==role[int(bb)]
 role.append(role[int(aa)])
role=np.asarray(role);outside=u[count==1]
edges=np.sort(np.concatenate([outside[:,[0,1]],outside[:,[1,2]],outside[:,[2,0]]]),axis=1)
_,ec=np.unique(edges,axis=0,return_counts=True)
checks={'status':'conforming_refinement_topology_control','face_incidence_gt2':int(np.count_nonzero(count>2)),'paired_face_orientation_errors':int(np.count_nonzero((count==2)&(balance!=0))),'closed_outer_volume_edge_errors':int(np.count_nonzero(ec!=2)),'shared_skin_fat_interface_faces':int(shared.sum()),'shared_interface_non_I_faces':int(np.count_nonzero(np.any(role[u[shared]]!=1,axis=1))),'source_original_labels_inherited':True,'physical_admission':False}
(out/'cheap-topology.json').write_text(json.dumps(checks,indent=2)+'\n')
assert not any(checks[k] for k in ['face_incidence_gt2','paired_face_orientation_errors','closed_outer_volume_edge_errors','shared_interface_non_I_faces']),checks
candidate=out/'unprojected-refined-source-control.npz'
np.savez_compressed(candidate,points_reference_world_m_f64=x.astype(float),volume_tetrahedra=t,material_index=mat,child_parent_tet_row=cp,child_parent_measure_fraction=r['child_source_measure_fractions'],geometric_donor_body=a['geometric_donor_body'][cp],anatomical_region_id=a['anatomical_region_id'][cp])
OWNER=Path('/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src')
BASE=OWNER.parents[1]/'incremental-exact-audit-owner-253/src';sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('numilab_human.refinement_exact531',OWNER/'numilab_human/cardiac_cavity_intersections.py');exact=importlib.util.module_from_spec(spec);sys.modules[spec.name]=exact;spec.loader.exec_module(exact)
sp=importlib.util.spec_from_file_location('gate531',P/'audit_coarse_material_bone_volume_427.py');gate=importlib.util.module_from_spec(sp);sp.loader.exec_module(gate)
layers={};start=time.monotonic()
for material,name in [(15,'skin'),(16,'fat')]:
 faces=gate.exterior(t[mat==material]);records,_=exact.float32_surface_records(x,faces,source_seam_groups=())
 result=exact._audit_pair(records,records,same_surface=True)
 layers[name]={'faces':len(faces),'self_intersection_pairs':len(result['triangle_pairs']),'audit':result}
 print(json.dumps({'layer':name,'faces':len(faces),'self_pairs':len(result['triangle_pairs'])}),flush=True)
receipt={'status':'exact_refined_material_boundary_self_audit','candidate_sha256':sha(candidate),'refinement_source_sha256':sha(P/'conforming-refinement-control-530/run-001/refinement-connectivity.npz'),'query_owner_sha256':sha(exact.__file__),'coordinate_contract':'canonical Float32 metre coordinates; exact integer predicates; no seam exemptions','layers':layers,'elapsed_s':time.monotonic()-start,'physical_admission':False,'runtime':0,'accepted_steps':0,'scope':'unprojected source002 refinement control only; no bone-clearance or anatomy qualification'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='layers'}),flush=True)
