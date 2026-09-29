from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent;repo=root.parents[1];out=root/'source-history';out.mkdir(exist_ok=True)
def save(name,text,wanted):
 data=text.encode();assert hashlib.sha256(data).hexdigest()==wanted,name;(out/name).write_bytes(data)
initial=json.loads((root/'tests.final/executed-source.json').read_text());retry=json.loads((root/'tests.retry/executed-source.json').read_text())
module=(repo/'src/numilab_human/organ_family_geometry.py').read_text()
block="""    lock = human.read_json(ROOT / 'sources.lock.json')['sources']['bodyparts3d_4']['files']
    for filename in ['isa_BP3D_4.0_obj_99.zip', 'partof_BP3D_4.0_obj_99.zip']:
        path = sources / filename
        require(path.is_file() and path.stat().st_size == lock[filename]['bytes']
                and human.sha256(path) == lock[filename]['sha256'], 'source archive identity: ' + filename)
"""
assert block in module;module=module.replace(block,'',1)
save('organ-family-before-early-archive-check.py',module,initial['src/numilab_human/organ_family_geometry.py'])
current=(repo/'tests/test_organ_family_geometry.py').read_text()
start=current.index("@pytest.mark.parametrize('changed'");end=current.index("@pytest.mark.parametrize('corruption',['missing'",start)
previous=current[:start]+current[end:]
previous=previous.replace("('ABI1','torso-organ-coverage-20260929/current-neutral.v2')","('ABI1','torso-organ-coverage-20260929/current-neutral')",1)
save('tests-first-retry.py',previous,retry['tests/test_organ_family_geometry.py'])
old=previous.replace("    assert selection['baseline_source_members']['FJ1932']['myosim_body']=='Abdomen'", "    assert next(r for r in added if r['member_id']=='FJ1932')['myosim_body']=='Abdomen'",1)
old=old.replace("'owner':(24,20 if p[6]==7 else 7)","'owner':(24,7)",1)
old=old.replace("    elif field in ['owner','baseline']:\n        row=310 if field=='owner' else 0\n        struct.pack_into('<I',raw,60+32*row,20 if records[row,0]==7 else 7)","    elif field in ['owner','baseline']:struct.pack_into('<I',raw,60+32*(310 if field=='owner' else 0),7)",1)
old=old.replace("    if '--focus-body' in cmd:cmd[cmd.index('--focus-body')]='--focus-body-index'\n",'',1)
save('tests-initial.py',old,initial['tests/test_organ_family_geometry.py'])
print('All three historical source snapshots match their captured execution hashes')
