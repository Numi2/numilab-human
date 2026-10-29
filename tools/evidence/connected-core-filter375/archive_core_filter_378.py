from pathlib import Path
import hashlib,json,shutil,tarfile
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');OUT=P/'core-checkpoint-378';PAY=OUT/'payload';PAY.mkdir(parents=True,exist_ok=False);C=P/'deep-core-connected-patch-filter-375';S=P/'deep-core-void-coalescence-366'
for n in ['LICENSE','THIRD_PARTY_NOTICES.md']:shutil.copy2(P/'core-checkpoint-366/payload'/n,PAY/n)
for n in ['candidate-f32-typed.npz','D-closed-csg-only.npz','D-derived-source-62000200.npz','geometry-checkpoint.npz','vertex-compaction-map.npz','receipt.json']:shutil.copy2(C/n,PAY/n)
for n in ['candidate-f32-typed.npz','D-closed-csg-only.npz']:shutil.copy2(S/n,PAY/('parent366-'+n))
shutil.copy2(P/'deep-core-change-coverage-377/receipt.json',PAY/'final-change-domain-and-local-self-gate.json')
shutil.copy2(P/'deep-core-change-coverage-374/receipt.json',PAY/'unfiltered-positive-occupancy-witnesses.json')
for n in ['batch_deep_core_patch_remesh_371.py','filter_core_patch_operations_375.py','check_filtered_core_candidate_377.py']:shutil.copy2(P/n,PAY/n)
(PAY/'README.txt').write_text('Connected excluded-core checkpoint375 with final gate377. Mixed-source reference anatomy; not a measured person. The unfiltered91-patch proposal introduced tiny occupied-domain losses. All50 patches overlapping all185 bank loss witnesses were removed, retaining41 patches. The filtered new faces have0 exact local self pairs against final topology and0 added exposed-bone volume in the F64 changed-domain Boolean diagnostic for185 banks. This does not certify parent whole-body containment, thickness, volume cells, mass transfer, physical steps or motion. Parent366 remains archived at the linked release. Source scripts retain absolute owner paths; replay_archive.py independently checks restored bytes, retained coordinates, exact unchanged-face ancestry, cap geometry and declared final gates using NumPy. It does not rerun original exact-predicate/Boolean programs or the simulator.\n')
(PAY/'replay_archive.py').write_text('''from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent
m=json.loads((P/'manifest.json').read_text())
for name,want in m['files'].items():assert hashlib.file_digest((P/name).open('rb'),'sha256').hexdigest()==want,name
def load(name):
 with np.load(P/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}
a=load('parent366-candidate-f32-typed.npz');n=load('candidate-f32-typed.npz');g=load('geometry-checkpoint.npz');v=load('vertex-compaction-map.npz');closed=load('D-closed-csg-only.npz');parentclosed=load('parent366-D-closed-csg-only.npz');keep=g['kept_input_face_rows'];count=len(keep)
assert np.array_equal(n['points_world_m'],a['points_world_m'][v['new_to_old']])
assert np.array_equal(n['points_world_m'][n['faces'][:count]],a['points_world_m'][a['faces'][keep]])
for k,val in a.items():
 if k.startswith('face_source_') and val.ndim and len(val)==len(a['faces']):assert np.array_equal(n[k][:count],val[keep]),k
po=a['face_ancestry_offsets'];no=n['face_ancestry_offsets'];take=np.concatenate([np.arange(po[i],po[i+1]) for i in keep])
assert np.array_equal(np.diff(no)[:count],np.diff(po)[keep])
for k in [k for k in a if k.startswith('face_ancestry_') and k!='face_ancestry_offsets']:assert np.array_equal(n[k][:no[count]],a[k][take]),k
assert np.array_equal(closed['points_world_m'][closed['faces'][-41:]],parentclosed['points_world_m'][parentclosed['faces'][-41:]])
r=json.loads((P/'final-change-domain-and-local-self-gate.json').read_text());assert r['manifest_banks_checked']==185 and r['sum_lost_core_intersect_changed_bones_m3']==0 and r['new_face_exact_pair_count_against_final_filtered_surface']==0
print(json.dumps({'status':'PASS_RESTORED_BYTES_COORDINATES_ANCESTRY_CAPS','files_verified':len(m['files']),'retained_faces':count,'new_faces':len(n['faces'])-count,'physical_admission':False,'original_predicates_rerun':False},indent=2))
''')
files={f.name:hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in sorted(PAY.iterdir()) if f.is_file()};(PAY/'manifest.json').write_text(json.dumps({'files':files,'physical_admission':False},indent=2)+'\n');archive=OUT/'human-core-connected-filter375.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as t:t.add(PAY,arcname='human-core-connected-filter375')
sha=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest();(OUT/(archive.name+'.sha256')).write_text(sha+'  '+archive.name+'\n');r={'archive':str(archive),'sha256':sha,'bytes':archive.stat().st_size,'manifest_files':len(files),'physical_admission':False};(OUT/'archive-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
