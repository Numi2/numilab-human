"""Check which composed-model inputs agree byte-for-byte with the pinned source archive."""
from pathlib import Path
import sys,hashlib,json,tarfile
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(archive,checkout,out):
 assert sha(archive)=='280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975'
 rows=[];changed=[]
 with tarfile.open(archive,'r:gz') as t:
  for member in t:
   if not member.isfile():continue
   name=Path(*Path(member.name).parts[1:])
   if not name.parts or name.parts[0]!='myo_sim' or name.suffix.lower() not in ('.xml','.stl','.obj','.py','.msh'):continue
   source=hashlib.sha256(t.extractfile(member).read()).hexdigest()
   p=checkout/name;actual=sha(p) if p.is_file() else None
   rows.append((str(name),source,actual))
   if source!=actual:changed.append(rows[-1])
 assert rows
 arm=[r for r in rows if r[0].startswith(('myo_sim/models/arm/','myo_sim/build/'))]
 required={'myo_sim/models/arm/assets/myoarm_r_tendon.xml','myo_sim/models/arm/assets/myoarm_r_chain.xml','myo_sim/build/compose.py'}
 assert required<={n for n,a,b in arm if a==b}
 out.write_text(json.dumps(dict(archive_sha256=sha(archive),checked_source_files=len(rows),arm_and_composition_files=len(arm),
       inventory_digest=hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest(),changed=changed),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
