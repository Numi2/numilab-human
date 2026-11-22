"""Recover old native viewer products from GitHub before retiring their local storage."""
import json,sys,shutil,stat
from pathlib import Path
base,source,active,out=map(lambda x:Path(x).resolve(),sys.argv[1:])
out.mkdir(exist_ok=True)
roots=['resting-clearance-and-atrium-'+i for i in ('002','003','004','005','006','007','008','009','010','011','012','013','014','015','016')]
protected=active.read_text()
plan=out/'selected-paths.json'
if not plan.exists():
    chosen=[]
    for key in roots:
        for p in sorted((base/key).rglob('*.mrvpack.zst')):
            if not (p.parent.name=='accepted-geometry' or (p.name=='resting-human.mrvpack.zst' and p.parent.name=='native')):continue
            st=p.lstat()
            if str(p) in protected or p.is_symlink() or not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_blocks*512<40<<20:continue
            chosen.append(str(p))
    plan.write_text(json.dumps(chosen,indent=2)+'\n')
files=list(map(Path,json.loads(plan.read_text())))
s=source.read_text();body=s[s.index("repo='Numi2/"):].replace("repo='Numi2/macmini-evidence-archive';tag='numi-20261104-cold-human-evidence'","repo='Numi2/numilab-human';tag='human-closed-skin-diagnostic-packs-20261121'")
for start in range(0,len(files),3):
    label='historic-view-volume-before-seated-solver-20261122-'+str(start//3)
    folder=out/label
    if (folder/(label+'-verification.json')).exists():continue
    if shutil.disk_usage(out).free>5*1024**3:break
    batch=files[start:start+3]
    assert all(p.is_file() and p.stat().st_nlink==1 and str(p) not in protected for p in batch)
    folder.mkdir(exist_ok=False)
    exec(compile(s[:s.index('W=Path(')]+body.replace('human-cold-motion-packs-320',label),'verified_native_view_archive','exec'),dict(W=base,D=folder,files=batch))
