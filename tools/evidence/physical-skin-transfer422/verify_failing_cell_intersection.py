from pathlib import Path
import json,numpy as np,sys,hashlib
R=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207');sys.path.insert(0,str(R/'authoring-deps'));import igl,manifold3d as mf
P=R/'physical-transfer-209';D=Path(__file__).resolve().parent
candidate=P/'reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz'
with np.load(candidate) as z:points=z['points_reference_world_m_f64'][z['volume_tetrahedra'][26104]]
faces=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.uint64)
def solid(v,f):return mf.Manifold(mf.Mesh64(vert_properties=np.ascontiguousarray(v,np.float64),tri_verts=np.ascontiguousarray(f,np.uint64)))
cell=solid(points,faces);assert cell.volume()>0
manifest=json.loads((P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json').read_text())
q=np.vstack([points,points.mean(0)]);results=[]
for bank in manifest['operands']:
 if (bank['body'],bank['bone']) not in [(109,114),(107,108)]:continue
 f=Path(bank['operand_path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==bank['operand_sha256']
 with np.load(f) as z:v=z['points_world_m_f64'];t=z['faces_f64'].astype(np.int64)
 w=igl.winding_number(v,t,q)
 inter=cell^solid(v,t)
 results.append({'body':bank['body'],'bone':bank['bone'],'operand_sha256':bank['operand_sha256'],'winding_at_vertices_then_centroid':w.tolist(),'cell_intersection_m3':inter.volume(),'intersection_fraction_of_tet':inter.volume()/cell.volume(),'boolean_status':str(inter.status())})
receipt={'scope':'local coarse physical penetration localization; not global geometry admission','candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'local_tet_row':26104,'global_tet_row':29419,'cell_volume_m3':cell.volume(),'banks':results}
(D/'failing-cell-verified-bone-intersections.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
