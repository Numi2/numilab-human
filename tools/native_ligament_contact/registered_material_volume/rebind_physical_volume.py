"""Rebind the retained fine P1 and O/I/D material ancestry to the freshly compiled native parent."""
from pathlib import Path
import sys,json,os,subprocess
import numpy as np
from emit_skin_source_registration import sha
baseline,reg,out=map(Path,sys.argv[1:]);capt=reg/'current-parent-capture';previous=baseline/'refit-motion-qualification'
out.mkdir(exist_ok=True);dest=out/'p1';dest.mkdir(exist_ok=False)
prior=previous/'p1/current-paired-volume-state.json';state=json.loads(prior.read_text())
base=capt/'compiled-parent-cell-mass.csv';receipt=json.loads((capt/'receipt.json').read_text())
fields={'source330_mass':'','source330_nodes':'.nodes.csv','source330_tets':'.tets.csv','source330_labels':'.labels.csv','source330_contributions':'.source.csv'}
live={k:Path(str(base)+s) for k,s in fields.items()}
for path in live.values():assert sha(path)==receipt['outputs'][path.name]
parent=state['historical_preference_source330']
def table(k,columns):
    h=parent[k];p=Path(h['path']);assert sha(p)==h['sha256']
    return [np.loadtxt(f,delimiter=',',skiprows=1,usecols=columns) for f in (p,live[k])]
old,new=table('source330_nodes',(0,1,2,3,4,9));pt,now=table('source330_tets',range(7));assert np.array_equal(old[:,[0,1,5]],new[:,[0,1,5]]) and np.array_equal(pt,now)
old_labels,labels=table('source330_labels',range(5));assert np.array_equal(old_labels[:,:4],labels[:,:4])
frame_delta=np.linalg.norm(old[:,2:5]-new[:,2:5],axis=1);t_delta=np.max(frame_delta[now[:,3:7].astype(int)],axis=1)
state.update(source330_identity={k:dict(path=str(p),sha256=sha(p))for k,p in live.items()},live_parent_capture_sha256=sha(capt/'receipt.json'),p1_current_parent_maximum_displacement_m=float(frame_delta.max()))
for role in ('skin','fat'):
    rec=json.loads((previous/f'p1/{role}-p1-receipt.json').read_text());p=Path(rec['p1_path']);assert sha(p)==rec['p1_sha256']
    with np.load(p) as packet:
        owners=np.repeat(np.arange(len(packet['node_group_offsets'])-1),np.diff(packet['node_group_offsets']))
        owners=np.repeat(owners,np.diff(packet['node_group_parent_tet_offsets']))
        d=packet['node_parent_registration_distance_m'][owners]+t_delta[packet['node_group_parent_tet_rows']]
        if not np.all(np.isfinite(d)&(d<=state['p1_current_distance_limit_m'])):raise ValueError('P1 ancestry exceeds retained distance '+role)
        state[f'p1_{role}_current_distance_upper_bound_m']=float(d.max())
state_path=dest/prior.name;state_path.write_text(json.dumps(state,indent=2)+'\n')
for role in ('skin','fat'):
    rec=json.loads((previous/f'p1/{role}-p1-receipt.json').read_text());old_base=rec['p1_sha256'];dept=dest/role.title();dept.mkdir()
    for pk,hk in (('p1_path','p1_sha256'),('p1_one_hop_path','p1_one_hop_sha256')):
        src=Path(rec[pk]);assert sha(src)==rec[hk]
        with np.load(src) as pack:data=dict(pack)
        assert str(data['current_state_sha256'])==sha(prior) and str(data['source330_mass_sha256'])==json.loads(prior.read_text())['source330_identity']['source330_mass']['sha256']
        data.update(current_state_path=np.asarray(str(state_path)),current_state_sha256=np.asarray(sha(state_path)),source330_mass_sha256=np.asarray(sha(base)))
        if pk=='p1_one_hop_path':
            assert str(data['admissibility_source_support_sha256'])==old_base
            data['admissibility_source_support_sha256']=np.asarray(rec['p1_sha256'])
        target=dept/src.name;np.savez_compressed(target,**data);rec[pk],rec[hk]=str(target),sha(target)
    rec.update(state_path=str(state_path),state_sha256=sha(state_path));(dest/f'{role}-p1-receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
# Reuse the production preparation algorithm to recompute every current-space
# barycentric residual for the exact same fine vertices and native parent.
script=baseline/'rebind_live_parent_registration.py';code=script.read_text();a=code.index('root = Path(');b=code.index('state = json.loads(',a)
code=code[:a]+'''root=baseline
prior=previous/'registration'
out=destination/'registration'
out.mkdir()
state_path=destination/'p1/current-paired-volume-state.json'
'''+code[b:]
code=code.replace('"absolute-volume-repaired-tetgen/mesh.node"','"constitutive-witness-repaired-tetgen/mesh.node"').replace('["/opt/homebrew/bin/zstd", "-dc", prior / "combined-node-map.csv.zst"]','["/bin/cat", prior / "combined-node-map.csv"]')
exec(compile(code,str(script),'exec'),dict(baseline=baseline,previous=previous,destination=out))
