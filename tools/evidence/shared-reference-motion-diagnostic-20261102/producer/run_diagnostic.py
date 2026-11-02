import csv,hashlib,json,struct,subprocess
from pathlib import Path
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
R=Path('/Users/numi/Work/skin-fat-native-integration-209')
O=H/'shared-reference-motion-diagnostic-root-001'
def pin(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
rows=set()
for name in ['translate_z_plus_1mm','rotate_z_minus_1deg','rotate_z_plus_1deg']:
 p=H/'mass-aware-inner-I-review-002/1020_original'/name/'tet-regressions.csv'
 with p.open()as f:
  for x in csv.DictReader(f):rows.add(int(x['tet_row']))
rows=np.array(sorted(rows),dtype=np.int64)
mesh=H/'current-native-1024-root/current-433-1044/volume-export/volume'
p=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3),dtype=np.float64).astype(np.float32)
t=np.loadtxt(mesh/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype=np.int64)-1
if len(p)!=311768 or len(t)!=1495294:raise ValueError('domain changed')
fixture=O/'selected-rest-f32.bin'
with fixture.open('wb')as f:f.write(struct.pack('<Q',len(rows)));f.write(p[t[rows]].astype('<f4').tobytes())
np.savetxt(O/'selected-tet-rows.csv',rows,fmt='%d',header='tet_row',comments='')
argv=['/usr/bin/clang++','-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(R/'include'),'-I'+str(R/'matter/include'),str(O/'shared_reference_motion.cpp'),'-o',str(O/'shared-reference-motion')]
b=subprocess.run(argv,capture_output=True,text=True);(O/'build.stdout').write_text(b.stdout);(O/'build.stderr').write_text(b.stderr)
if b.returncode:raise RuntimeError('compile failed '+b.stderr)
r=subprocess.run([str(O/'shared-reference-motion'),str(fixture)],capture_output=True,text=True)
(O/'run.stdout').write_text(r.stdout);(O/'run.stderr').write_text(r.stderr)
if r.returncode:raise RuntimeError('math check failed '+r.stderr)
d=json.loads(r.stdout)
out={'status':'COMPLETED_SHARED_ARITHMETIC_DIAGNOSTIC_NO_NATIVE_EXECUTION','physical_admission':False,'native_steps':0,'gpu_executed':False,'selected_rows':len(rows),'selection':'union of all original1020 failures in the three materialized-F32 rigid-frame probes, not unbiased fullmesh','inputs':{k:pin(v)for k,v in [('mesh_node',mesh/'mesh.node'),('mesh_ele',mesh/'mesh.ele'),('cpp',O/'shared_reference_motion.cpp'),('fixture',fixture),('rows',O/'selected-tet-rows.csv'),('shared_header',R/'matter/include/numi/matter/fem_compensated_deformation.h'),('compensated_header',R/'include/metalrobo/compensated_geometry_gpu.h'),('translation_header',R/'include/metalrobo/compensated_translation_gpu.h')]},'compile_argv':argv,'arithmetic':d,'limitations':['CPU instantiation of the actual shared Metal/C++ arithmetic helper, not GPU execution','synthetic exact rigid displacement split into F32 high/low; no forces/contact or controller advanced','source cells and inverseRest casting mirror compiler; actual GPU state still requires native verification']}
(O/'diagnostic-report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'report':pin(O/'diagnostic-report.json'),'arithmetic':d},indent=2))

