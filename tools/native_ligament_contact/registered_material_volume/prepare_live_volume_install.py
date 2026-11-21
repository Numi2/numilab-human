"""Select only the physical parent, registration and allocation proven by native replay."""
from pathlib import Path
import json,sys
from emit_skin_source_registration import sha
original,baseline,reg,output=map(Path,sys.argv[1:])
launch=json.loads(original.read_text());fine=baseline/'refit-motion-qualification';new=reg/'material';cap=reg/'current-parent-capture'
proof=json.loads((new/'native-role-verification.json').read_text())
if not proof['passed']:raise ValueError('Both native role proofs are required')
pairs={}
def track(old,current):
    if not old.is_file() or not current.is_file():raise ValueError((old,current))
    pairs[str(old.resolve())]=str(current.resolve())
    a,b=sha(old),sha(current)
    if a in pairs and pairs[a]!=b:raise ValueError('Ambiguous transformation of original identity')
    pairs[a]=b
for p in cap.glob('compiled-parent-cell-mass.csv*'):track(baseline/'floor-live-parent-capture-002'/p.name,p)
for n in ('combined-node-map.csv','node-parent-registration.csv','facet-ancestry-current.csv','registered-O.csv','registered-I.csv','registered-D.csv','all-body-current-rest.csv'):track(fine/'registration'/n,new/'registration'/n)
track(fine/'p1/current-paired-volume-state.json',new/'p1/current-paired-volume-state.json')
a,b=[json.loads((w/'witnesses/receipt.json').read_text()) for w in (fine,new)]
for old,now in zip(a['roles'],b['roles']):
    assert old['role']==now['role']
    for k in ('witness_path','carrier_path'):track(Path(old[k]),Path(now[k]))
for role in ('skin','fat'):
    a,b=[json.loads((w/f'p1/{role}-p1-receipt.json').read_text()) for w in (fine,new)]
    for k in ('p1_path','p1_one_hop_path'):track(Path(a[k]),Path(b[k]))
for n,h in proof['artifacts'].items():
    path=new/'native-role-flow'/n
    if sha(path)!=h:raise ValueError(path)
    track(fine/'native-role-flow'/n,path)
active=launch['environment'];changes={}
for k,v in list(active.items()):
    if v in pairs and v!=pairs[v]:
        active[k]=pairs[v];changes[k]=dict(before=v,after=pairs[v])
launch['pins']={pairs.get(k,k):pairs.get(v,v)for k,v in launch['pins'].items()}
launch['selected_physical_mass']={'native_parent_capture_sha256':sha(cap/'receipt.json'),'solver_free_role_replay_sha256':sha(new/'native-role-verification.json'),'source_changes':changes}
output.write_text(json.dumps(launch,indent=2)+'\n')
