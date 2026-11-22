"""Bind a replayed connected material cavity to the exact previously exercised native launch."""
from pathlib import Path
import json, sys
from emit_skin_source_registration import sha

def main(source, old_material, inputs, output):
    launch = json.loads(source.read_text())
    env = launch['environment']
    material = inputs/'material'
    new = json.loads((material/'native-role-verification.json').read_text())
    registration = json.loads((material/'registration/receipt.json').read_text())
    binding = json.loads((inputs/'mesh/current-child.labels.json').read_text())
    state = sha(material/'p1/current-paired-volume-state.json')
    if not new['passed'] or state != binding['native_p1_state_sha256'] or state != registration['current_parent_state_sha256']:
        raise ValueError('Unmatched physical source revisions')
    prefix = 'NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def at_input(key, value):
        if key == prefix+'TETGEN_NODE': return inputs/'mesh/mesh.node'
        if key == prefix+'CHILD_LABELS': return inputs/'mesh/current-child.labels.csv'
        p = Path(value)
        if p.is_relative_to(old_material): return material/p.relative_to(old_material)
        return None
    changes = {}
    def update(key, value):
        if env[key] != value:
            changes[key] = (env[key], value)
            env[key] = value
    paths = {}
    hashes = {}
    for key, value in list(env.items()):
        target = at_input(key,value)
        if target is None: continue
        old = Path(value)
        if not target.is_file(): raise ValueError(str(target))
        before, after = sha(old), sha(target)
        if before not in env.values(): raise ValueError('Unpinned input '+key)
        if before in hashes and hashes[before] != after: raise ValueError('Ambiguous shared digest '+key)
        hashes[before] = after
        paths[value] = str(target)
        update(key, str(target))
    for role in ('skin','fat'):
        old, new_p1 = [json.loads((m/'p1'/f'{role}-p1-receipt.json').read_text()) for m in (old_material,material)]
        assert new_p1['state_sha256'] == state
        for key in ('state_sha256','role_geometry_sha256','p1_one_hop_sha256'):
            if old[key] in hashes and hashes[old[key]] != new_p1[key]: raise ValueError(key)
            hashes[old[key]]=new_p1[key]
    for key, value in list(env.items()):
        if value in hashes: update(key,hashes[value])
    for name, h in new['artifacts'].items():
        path = str(material/'native-role-flow'/name)
        if path not in env.values() or h not in env.values(): raise ValueError('Missing recorded owner '+name)
    for path in ('registered-O.csv','registered-I.csv','registered-D.csv','combined-node-map.csv','facet-ancestry-current.csv'):
        if registration['outputs'][path] not in env.values(): raise ValueError('Missing current surface '+path)
    launch['pins']={paths.get(k,k):hashes.get(v,v) for k,v in launch['pins'].items()}
    launch['physical_volume_release_reference'] = dict(launch_path=str(source), launch_sha256=sha(source))
    launch['connected_material_volume'] = dict(parent_launch_sha256=sha(source), source_sha256=registration['proposal_sha256'],
        role_reader_sha256=sha(material/'native-role-verification.json'), registration_sha256=sha(material/'registration/receipt.json'),
        current_state_sha256=state, changed_keys=sorted(changes))
    output.write_text(json.dumps(launch,indent=2)+'\n')
    print('bound',len(changes),'settings for',state)
if __name__=='__main__': main(*(Path(x).resolve() for x in sys.argv[1:]))
