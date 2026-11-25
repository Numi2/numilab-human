"""Bind the six newly audited material paths to the exact native start pose."""
from pathlib import Path
import shutil,json,subprocess,sys
from compile_connected_joint import digest
base,cert,out=(Path(p).resolve()for p in sys.argv[1:])
summary=json.loads((cert/'registered-fibers.json').read_bytes());inc=cert/'NumiHumanCorticalRegistration.inc'
if summary['candidate_inc']!=digest(inc) or summary['bone_count']!=185 or len(summary['paths'])!=6 or any(r['crossings'] for r in summary['paths']):raise ValueError('Unverified cortical release')
out.mkdir()
for p in base.iterdir():
 if p.is_file() and p.suffix not in ['.o','.air'] and p.name not in ['numi-human-native','commands.json']:shutil.copy2(p,out/p.name)
shutil.copytree(base/'metalrobo',out/'metalrobo',symlinks=True)
shutil.copy2(inc,out/inc.name)
cmds=json.loads((base/'commands.json').read_bytes())[-2:]
for c in cmds:
 for i,v in enumerate(c):
  if Path(v).parent==base:c[i]=str(out/Path(v).name)
 subprocess.run(c,cwd=Path.home()/'Work/skin-fat-native-integration-209/build-native',check=True)
(out/'commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
