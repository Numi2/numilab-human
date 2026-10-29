from pathlib import Path
import numpy as np,json,sys,importlib.util,hashlib,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
C=P/'reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz'
D=P/'conforming-fit-006-exact-self-549/run-001';D.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=np.load(C,allow_pickle=False);x=z['points_reference_world_m_f64'].astype(np.float32);t=z['volume_tetrahedra'];mat=z['material_index']
OWNER=Path('/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src')
BASE=OWNER.parents[1]/'incremental-exact-audit-owner-253/src';sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('numilab_human.refinement_exact549',OWNER/'numilab_human/cardiac_cavity_intersections.py');exact=importlib.util.module_from_spec(spec);sys.modules[spec.name]=exact;spec.loader.exec_module(exact)
sp=importlib.util.spec_from_file_location('gate549',P/'audit_coarse_material_bone_volume_427.py');gate=importlib.util.module_from_spec(sp);sp.loader.exec_module(gate)
layers={};start=time.monotonic()
for material,name in [(15,'skin'),(16,'fat')]:
 faces=gate.exterior(t[mat==material])
 records,_=exact.float32_surface_records(x,faces,source_seam_groups=())
 result=exact._audit_pair(records,records,same_surface=True)
 layers[name]={'faces':len(faces),'self_intersection_pairs':len(result['triangle_pairs']),'audit':result}
 print(json.dumps({'layer':name,'faces':len(faces),'self_pairs':len(result['triangle_pairs'])}),flush=True)
receipt={'status':'exact_refined_material_boundary_self_audit','candidate_sha256':sha(C),
 'query_owner_sha256':sha(exact.__file__),'script_sha256':sha(Path(__file__)),
 'coordinate_contract':'canonical Float32 metre coordinates; exact integer predicates; no seam exemptions',
 'layers':layers,'elapsed_s':time.monotonic()-start,'physical_admission':False,'runtime':0,'accepted_steps':0,
 'scope':'006 material self-check only; known bank23/29 skin/fat occupancy failure is not waived'}
(D/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='layers'}),flush=True)
