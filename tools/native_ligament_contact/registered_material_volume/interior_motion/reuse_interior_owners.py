"""Increment the existing material support and allocator for relocated interior quadrature sites."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest

def patched(source,destination,edits):
    text=source.read_text()
    for old,new,count in edits:
        if text.count(old)!=count:raise ValueError('Source drift in '+source.name+' at '+old[:64])
        text=text.replace(old,new)
    destination.write_text(text)
def run(w,parent,stage,volume):
    s=w/stage;s.mkdir(exist_ok=True)
    m=s/'witness-material';m.mkdir(exist_ok=True); base=parent/'witness-material'
    for target,path in [(m/'volume',w/volume),(s/'current-p1',base/'current-p1'),(s/'owner.locator',parent/'owner.locator'),
                        (s/'recorded-mass',base/'recorded-mass'),(s/'native-integrated-01',parent/'accepted-material-07')]:
        if not target.exists():target.symlink_to(path)
        if target.resolve()!=path.resolve():raise ValueError('Prior stage mismatch')
    s1=s/'rebuild_current_material_basis.py'
    patched(parent/s1.name,s1,[('        nd[-1]=-1','        # No new nodes; the owner detects changed Float32 positions by comparison.',1)])
    s2=s/'replay_native_credits.py'
    a=(parent/s2.name).read_text()
    start=a.index("            affected=overlay['old_cells']")
    stop=a.index('        else:\n            a=np.load(q/',start)
    replacement='''            affected=overlay['old_cells'][oldmat[overlay['old_cells']]==mat]
            if np.isin(comp,affected).any():raise ValueError('An historical compensation row is inside the cavity')
            base[np.isin(base,affected)]=-1
            select=np.isin(cc,affected)
            oldp,oldc,oldw=aa[select],cc[select],mw[select]
            ps=[aa[~select]];cs=[cc[~select]];ws=[mw[~select]]
            for u,v,fraction in zip(overlay['from_cell'],overlay['to_cell'],overlay['fraction']):
                if oldmat[u]!=mat:continue
                if oldmat[v]!=mat or fraction<=0:raise ValueError('Invalid ancestry')
                loc=oldc==u
                if not loc.any():raise ValueError('Missing source credit')
                ps.append(oldp[loc]);cs.append(np.full(loc.sum(),v,dtype=cc.dtype));ws.append(oldw[loc]*fraction)
            aa=np.concatenate(ps);cc=np.concatenate(cs);mw=np.concatenate(ws)
'''
    patched(parent/s2.name,s2,[(a[start:stop],replacement,1),
           ("oldvol=q/'qualified-material' if refined",f"oldvol=Path({str(base/'volume')!r}) if refined",1),
           ("p.parent/'posed-app'",f"Path({str((base/'recorded-mass/posed-app').resolve())!r})",1),
           ("if refined and mat==16:", "if refined and np.any(cells[overlay['new_cells'],4]==mat):",1),
           ("subset=dest[overlay['new_cells']]","subset=dest[overlay['new_cells'][cells[overlay['new_cells'],4]==mat]]",1)])
    for stage,name,outcome in [(s1,'p1',m/'current-p1/result.json'),(s2,'allocation',m/'recorded-mass/native-replay/result.json')]:
        if outcome.exists():
            if json.loads(outcome.read_bytes()).get('exit',0)!=0:raise ValueError('Previous owner rejection')
            continue
        with (s/(name+'.log')).open('w') as f:
            subprocess.run([sys.executable,stage,'--motion-witness'],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('w','parent'):p.add_argument(a,type=lambda x:Path(x).resolve())
    p.add_argument('stage');p.add_argument('volume')
    run(**vars(p.parse_args()))
