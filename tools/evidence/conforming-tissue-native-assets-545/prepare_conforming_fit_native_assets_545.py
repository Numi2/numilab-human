from pathlib import Path
import csv,hashlib,json,math,subprocess,time,numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
F=P/'reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006'
C=P/'native-physical-parent-mass-330/native-source-identity-control-003'
D=P/'conforming-fit-006-native-assets-545/run-001'
D.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
z=np.load(F/'candidate.npz',allow_pickle=False)
pn=rows(C/'compiled-parent-cell-mass.csv.nodes.csv');pt=rows(C/'compiled-parent-cell-mass.csv.tets.csv')
p=np.array([[float(r['rest_'+k+'_m']) for k in 'xyz'] for r in pn])
t=np.array([[int(r['parent_node'+str(k)]) for k in range(4)] for r in pt])
cp=z['child_parent_tet_row'].astype(np.int64);ct=z['volume_tetrahedra'].astype(np.int64)
x=z['points_reference_world_m_f64'].astype(np.float32).astype(np.float64)
dist=np.linalg.norm(x[ct].mean(1)-p[t].mean(1)[cp],axis=1)
bound=max(.0002,math.ceil(float(dist.max())/1e-6)*1e-6+1e-6)
script=P/'export_refined_coarse_registered_volume_433.py'
cmd=['/Users/numi/numi-human-prep-venv-20261005/bin/python',str(script),
 '--candidate-npz',str(F/'candidate.npz'),'--candidate-node',str(F/'candidate.node'),
 '--candidate-node-map',str(F/'coarse-node-to-fine-surface.csv'),
 '--inserted-node-ancestry',str(F/'inserted-node-ancestry.csv'),
 '--capture',str(C),'--out',str(D/'registered-assets'),
 '--maximum-mass-transport-distance-m',format(bound,'.17g')]
plan={'command':cmd,'export_script_sha256':sha(script),'candidate_sha256':sha(F/'candidate.npz'),
 'predeclared_transport_envelope_m':bound,'maximum_declared_source_centroid_distance_m':float(dist.max()),
 'mass_transport_rule':'ceil(max child-to-declared-source-centroid distance/1um)*1um+1um, floor200um; derived before native evaluation',
 'purpose':'existing CSV asset compiler and native integration regression only; known bank29 failure retained',
 'physical_admission':False,'runtime':0,'accepted_steps':0}
(D/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
start=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True)
(D/'export.stdout').write_text(r.stdout);(D/'export.stderr').write_text(r.stderr)
out={'exit_code':r.returncode,'wall_s':time.monotonic()-start,'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr.encode()).hexdigest(),'physical_admission':False}
(D/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
raise SystemExit(r.returncode)
