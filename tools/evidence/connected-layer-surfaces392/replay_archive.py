from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent
manifest=json.loads((P/'manifest.json').read_text())
for name,record in manifest['files'].items():
 path=P/name
 assert path.stat().st_size==record['bytes'],name
 assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==record['sha256'],name
def arrays(name):
 with np.load(P/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}
a=arrays('O/parent.npz');n=arrays('O/candidate.npz');g=arrays('O/geometry-checkpoint.npz');keep=np.ones(len(a['faces']),bool);keep[g['removed_face_rows']]=False;rows=np.flatnonzero(keep);count=len(rows)
assert np.array_equal(n['points_world_m'].view(np.uint32),a['points_world_m'].view(np.uint32))
assert np.array_equal(n['faces'][:count],a['faces'][rows])
for key,value in a.items():
 if key.startswith('face_source_') and value.ndim and len(value)==len(a['faces']):assert np.array_equal(n[key][:count],value[rows]),key
po=a['face_ancestry_offsets'];no=n['face_ancestry_offsets'];take=np.concatenate([np.arange(po[i],po[i+1]) for i in rows])
assert np.array_equal(np.diff(no)[:count],np.diff(po)[rows])
for key in [k for k in a if k.startswith('face_ancestry_') and k!='face_ancestry_offsets']:assert np.array_equal(n[key][:no[count]],a[key][take]),key
for label in ['O','I','D']:
 z=arrays(label+'/candidate.npz');p=np.asarray(z['points_world_m'],np.float32);f=z['faces'];assert np.isfinite(p).all();assert f.min()>=0 and f.max()<len(p)
 tri=p[f].astype(float);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);assert np.all(np.any(cross!=0,axis=1))
 if label!='D':
  r=json.loads((P/(label+'/exact-native-report.json')).read_text());assert r['physical_O_self_exact_triangle_pairs']==0 and r['bank_count']==185 and r['exact_crossing_pair_count']==0
  assert r['physical_O_sha256']==manifest['files'][label+'/candidate.npz']['sha256']
print(json.dumps({'status':'PASS_RESTORED_BYTES_COORDINATES_ANCESTRY_EMISSION_AND_PINNED_GATES','files_verified':len(manifest['files']),'physical_admission':False,'original_predicates_rerun':False},indent=2))
