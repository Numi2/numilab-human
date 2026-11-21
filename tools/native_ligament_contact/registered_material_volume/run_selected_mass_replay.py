"""Exercise the existing native role writer and both independent role readers."""
import json,sys,subprocess,os,re
from pathlib import Path
from emit_skin_source_registration import sha
base,reg=map(Path,sys.argv[1:]);d=reg/'material';p=d/'p1';s=p/'current-paired-volume-state.json';st=json.loads(s.read_text());roles=[json.loads((p/(k+'-p1-receipt.json')).read_text()) for k in ['skin','fat']]
w=json.loads((d/'witnesses/receipt.json').read_text());assert w['current_state_sha256']==sha(s)
flow=reg/'fine-mass-routed/current-density-closed-flow.npz';mc=json.loads((reg/'fine-mass-routed/mass-transfer.json').read_text());assert sha(flow)==mc['new_flow_sha256']==w['verified_flow_sha256']
mesh=base/'constitutive-witness-repaired-tetgen';ledger=st['source330_identity'];labels=base/'paired-floor-child.labels.csv';wall=json.loads((base/'paired-floor-wall-lineage.json').read_text());assert sha(labels)==wall['labels_sha256']
old=json.loads((base.parent/'captured-fat-sliver-495/proposal/compiler/mass-allocation/run-manifest.json').read_text())['argv']
proof=base/'validate_complete_fine_p1';args=[proof,ledger['source330_mass']['path'],ledger['source330_labels']['path'],mesh/'mesh.ele',sha(s),ledger['source330_mass']['sha256']]
for v in roles:args.extend([v['p1_one_hop_path'],v['p1_one_hop_sha256'],v['role_geometry_sha256']])
args.extend([sum(v['cells'] for v in roles),mesh/'mesh.node',ledger['source330_tets']['path'],ledger['source330_nodes']['path']])
for field in ('carrier','witness'):
    for v in w['roles']:args.extend([v[field+'_path'],v[field+'_sha256']])
moves=sum(int(n)for n in re.findall(r'augmentations=(\d+)',mc['solver_output']))+len(mc['qualified_additional_routes'])
args.extend([mc['distance_bound_m'],ledger['source330_tets']['sha256'],ledger['source330_nodes']['sha256'],labels,sha(labels),old[old.index('--source-body-count')+1],old[old.index('--coordinate-frame-sha256')+1],flow,sha(flow),moves]);args=list(map(str,args))
assert len(args)==34
directory=d/'native-role-flow';directory.mkdir(exist_ok=False)
env={'NUMI_HUMAN_NATIVE_ROLE_FLOW_OUTPUT':str(directory)}
(d/'native-role-launch.json').write_text(json.dumps(dict(argv=args,environment=env,executable_sha256=sha(proof)),indent=2)+'\n')
with (d/'native-role.log').open('w') as log:result=subprocess.run(args,env={**os.environ,**env},stdout=log,stderr=log)
log=(d/'native-role.log').read_text()
passed=result.returncode==0 and all(k in log for k in ('Production role-flow writer PASS','Production skin-role-flow reader PASS','Production fat-role-flow reader PASS'))
(d/'native-role-verification.json').write_text(json.dumps(dict(exit_code=result.returncode,passed=passed,output=[line for line in log.splitlines() if 'PASS' in line or 'rejected' in line.lower()],artifacts={p.name:sha(p)for p in directory.glob('*')if p.is_file()}),indent=2)+'\n')
if not passed:sys.exit(1)
