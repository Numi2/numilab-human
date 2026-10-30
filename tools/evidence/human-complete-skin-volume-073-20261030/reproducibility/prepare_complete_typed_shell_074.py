"""Prepare full existing-owner shell ancestry for validated prebuilt child meshes.
Geometry-only; no tetra generation, mass allocation or physical admission."""
from pathlib import Path
import hashlib,importlib.util,json,sys,time,shutil
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
H=P/'fine-physical-volume-706'
OUT=H/'complete-typed-shell-owner-074'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 if shutil.disk_usage(H).free<9*2**30:raise RuntimeError('bounded preparation reserve below9GiB')
 OUT.mkdir(exist_ok=False)
 t0=time.monotonic()
 cp=H/'verified-volume-preparation-010-merged038005'
 c=json.loads((cp/'canonical-D/canonical-D-validation.json').read_text())
 r=json.loads((cp/'rim/receipt.json').read_text())
 assert c['status']=='PASS_EXISTING_CANONICAL_D_OWNER'
 assert r['status']=='PASS_COLLAR_AND_PROTECTED149_REINDEX_SOURCE_ONLY'
 pins={**{role:item for role,item in c['typed_candidates'].items()},**r['outputs']}
 old=json.loads((H/'current-D-proof-015/plan-source004-020/manifest.json').read_text())['inputs']
 for key in ('parent_cells','parent_nodes','wall_parent_cells','wall_parent_nodes'):pins[key]=old[key]
 for k,item in pins.items():
  if sha(item['path'])!=item['sha256']:raise ValueError('changed input '+k)
 path=H/'current-D-proof-015/handoff-source004/run_native_domain_217_typedD_697.py'
 spec=importlib.util.spec_from_file_location('typed_shell074',path)
 a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
 m=a.load_typed_owner();m.configure_capacity_profile('full-reference-500k-nodes-2m-cells')
 with np.load(pins['collar']['path'],allow_pickle=False) as z: collar={k:z[k] for k in z.files}
 O=m.load_surface(Path(pins['O']['path']),'O',collar)
 I=m.load_surface(Path(pins['I']['path']),'I')
 D=m.load_surface(Path(pins['D']['path']),'D')
 wall=m.read_wall_provenance(collar,Path(pins['wall_csv']['path']),
     Path(pins['wall_parent_cells']['path']),Path(pins['wall_parent_nodes']['path']),'registered-native-current')
 skin,fat,assembly=m.assemble_shells(O,I,D,collar,wall)
 parity={}
 for label,shell in [('skin',skin),('fat',fat)]:
  with np.load(cp/'plcs'/(label+'-shell.npz'),allow_pickle=False)as z:
   parity[label]={k:bool(np.array_equal(shell[k],z[k]))for k in ('points_world_m','faces','face_marker')}
  if not all(parity[label].values()):raise ValueError('typed owner geometry differs '+label)
  m.write_shell_npz(OUT/(label+'-shell.npz'),shell,label)
  m.write_face_ancestry_csv(OUT/(label+'-face-ancestry.csv.gz'),shell)
 shared=bool(np.array_equal(skin['points_world_m'][skin['registered_I_shell_node_rows']],fat['points_world_m'][fat['registered_I_shell_node_rows']]))
 if not shared:raise ValueError('shared I source discrepancy')
 report={'status':'PASS_EXISTING_OWNER_FULL_TYPED_SHELL_ANCESTRY_GEOMETRY_PARITY',
 'inputs':pins,'owner_sha256':sha(H/'native-domain-217-source-004/build_native_domain_217.py'),
 'adapter_runner_sha256':sha(path),'assembly':assembly,'geometry_exact_to_meshed_PLC':parity,
 'shared_I_exact':shared,'physical_admission':False,'runtime':0,'accepted_steps':0,
 'elapsed_s':time.monotonic()-t0,
 'outputs':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size}for p in OUT.iterdir() if p.is_file()}}
 (OUT/'receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'elapsed_s':report['elapsed_s'],'output':str(OUT)}),flush=True)
if __name__=='__main__':main()
