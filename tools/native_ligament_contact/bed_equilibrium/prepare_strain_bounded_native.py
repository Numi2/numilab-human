"""Bind the same native owner and source cells to an independently recovered seated pose."""
import json,shutil,subprocess,sys
from pathlib import Path
from evaluate_registered_bed_transfer import digest
base,physical,controls,pose,reg,out=map(lambda x:Path(x).resolve(),sys.argv[1:7])
previous=json.loads((base/'identities.json').read_text())
old=json.loads((controls/'loaded-registration.json').read_text());new=json.loads((pose/'loaded-registration.json').read_text())
assert old['source']==new['source'] and old['cell_count']==new['cell_count'] and old['node_count']==new['node_count']
assert len(new['attachment_rows'])==14 and new['largest_attachment_error_m']<=2e-7 and not new['outer_rows_below_plane']
a=json.loads((controls/'recovery.json').read_text());b=json.loads((pose/'recovery.json').read_text())
libid=json.loads((physical/'identities.json').read_text())
assert digest(physical/'libmetalrobo.dylib')==libid['sha256']['libmetalrobo.dylib']
out.mkdir(exist_ok=False)
for f in base.iterdir():
    if f.is_dir() and not f.is_symlink():continue
    if f.suffix in ('.hpp','.h','.inc','.metal','.metalinc','.mm','.metallib') or f.name=='metalrobo':
        if f.name=='NumiHumanLoadedSkinPoseData.inc':continue
        if f.is_symlink():(out/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
        else:(out/f.name).hardlink_to(f)
name='NumiHumanLoadedSkinPoseData.inc'
header=(base/name).read_text()
assert header.count(a['decoded_node_sha256'])==1 and digest(base/name)==previous['generated_header']
(out/name).write_text(header.replace(a['decoded_node_sha256'],b['decoded_node_sha256']))
if len(sys.argv)==8:
    from bind_barrier_initialization import bind
    bind(out,base,json.loads(Path(sys.argv[7]).read_text()),pose)
cmd=[[v.replace(str(base)+'/',str(out)+'/') for v in c] for c in previous['commands']]
environment=json.loads((physical/'config.json').read_text())['environment']
cmd[1][cmd[1].index('lib/libmetalrobo.dylib')]=str(physical/'libmetalrobo.dylib')
original=next(v for v in cmd[1] if v.endswith('/app-runtime.mm.o'))
cmd[1][cmd[1].index(original)]=str(physical/'app-runtime.mm.o')
for c in cmd:subprocess.run(c,cwd=reg/'build-native',check=True)
(out/'identities.json').write_text(json.dumps({'commands':cmd,'original_source':previous['source'],'header_sha256':digest(out/name),
    'pose_sha256':b['decoded_node_sha256'],'binary_sha256':digest(out/'numi-human-native'),'lib_sha256':digest(physical/'libmetalrobo.dylib')},indent=2)+'\n')
