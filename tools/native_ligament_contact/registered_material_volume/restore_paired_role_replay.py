"""Restore the tested native paired-role mass proof to the current physical constructor."""
from pathlib import Path
import sys,json,subprocess,shutil
from emit_skin_source_registration import sha
src,floor,out,app=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
for f in src.iterdir():
 if f.is_file() and f.suffix in ('.hpp','.metalinc','.metal') and not (out/f.name).exists():(out/f.name).hardlink_to(f)
(out/'metalrobo').mkdir(exist_ok=True)
for f in (src/'metalrobo').glob('*.metallib'):
 if not (out/'metalrobo'/f.name).exists():(out/'metalrobo'/f.name).hardlink_to(f)
names=['ConservativeSpatialSourceCarrierQuadrature.hpp','NumiHumanCurrentFractionalRecordedFlowBinding.hpp','NumiHumanCurrentFractionalSupportNPZ.hpp']
for n in names:
 p=out/(n+'.tmp');shutil.copyfile(floor/n,p);p.replace(out/n)
# Quoted transitive includes must resolve all three versioned owners together.
# Inherit the unchanged headers by hard link so the enclosing checkout stays intact.
for node in (app/'apps').glob('*.hpp'):
    target=out/node.name
    if not target.exists():target.hardlink_to(node)
n='numilab_human_myosim_visual_probe.mm'
def span(s):
 a=s.index('                CurrentFractionalRecordedFlowBinding recordedFlowBinding;');b=s.index('            } else',s.index('"registered_volume_current_fractional_recorded_carrier_allocation=',a))
 return a,b
s=(src/n).read_text();f=(floor/n).read_text();a,b=span(s);l,r=span(f)
if 'directRoleReplay' in s[a:b] or 'source_proof=paired_role_replay' not in f[l:r]:raise ValueError('Unrecognized recorded-flow source revision')
(out/n).write_text(s[:a]+f[l:r]+s[b:])
cmd=[[a.replace(str(src)+'/',str(out)+'/') for a in row] for row in json.loads((src/'build-commands.json').read_text())]
(out/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
for row in cmd:subprocess.run(row,cwd=app/'build-native',check=True)
(out/'identity.json').write_text(json.dumps({str(p):sha(p) for p in [src/n,floor/n,out/n,*[out/k for k in names],out/'numi-human-native',*(out/'metalrobo').glob('*.metallib')]},indent=2)+'\n')
