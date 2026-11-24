"""Run the current P1 compiler with the verified survivor map and replay source credits."""
import json,os,subprocess,sys
from pathlib import Path
from build_bed_audit import sha,replace_once
def link(target,source):
    if target.exists():
        if target.resolve()!=source.resolve():raise ValueError('Wrong input '+str(target))
    else:target.symlink_to(source.resolve())
def prepare(parent,case,mesh,ledger,out):
    prior=parent/'witness-material';vol=mesh/'volume';out.mkdir(exist_ok=True);s=out/'witness-material';s.mkdir(exist_ok=True)
    link(out/'owner.locator',parent/'owner.locator');link(out/'current-p1',prior/'current-p1');link(s/'volume',vol);link(mesh/'material-p1',s/'current-p1')
    rec=json.loads((mesh/'result.json').read_bytes());essential=('mesh.node','mesh.ele','current.node','cell-mass.f64','node-mass.f64','material-walls.npz','current-child.labels.csv')
    cert=dict(exit=0,native=rec['native'],inputs={k:sha(vol/k)for k in essential});(vol/'result.json').write_text(json.dumps(cert,indent=2)+'\n')
    code=(parent/'rebuild_current_material_basis.py').read_text();before="""        overlay=np.load(v/'source-overlay.npz')
        priorCells=np.arange(len(t),dtype='i4')
        priorCells[overlay['new_cells']]=-1"""
    after="""        mapping=np.load(v/'overlay.npz')['old_to_new']
        surviving=np.flatnonzero(mapping>=0)
        priorCells=np.full(len(t),-1,dtype='i4')
        priorCells[mapping[surviving]]=surviving"""
    script=out/'rebuild_current_material_basis.py';script.write_text(replace_once(code,before,after))
    subprocess.run([sys.executable,script,'--motion-witness'],check=True)
    job=out/'allocator';job.mkdir(exist_ok=True);(job/'verified-connected-material').mkdir(exist_ok=True)
    link(job/'verified-connected-material/scene',case);link(job/'deep-fat-ledger',prior/'recorded-mass');link(job/'posed-app',prior/'recorded-mass/posed-app')
    old=job/'deep-fat-native';old.mkdir(exist_ok=True)
    for name in ('cell-mass.f64','mesh.ele'):link(old/name,prior/'volume'/name)
    (old/'result.json').write_text(json.dumps(dict(native_command=rec['native_command'])))
    # The labels supplied by the geometric owner must also pass source P1 admission.
    label_sha=sha(vol/'current-child.labels.csv')
    code=ledger.read_text()
    # Skin has zero affected source rows in this transfer. An empty comprehension
    # needs an explicit boolean dtype to combine it with the distance predicate.
    token='good=np.array([rr[a] in rids[off[b]:off[b+1]] for a,b in zip(q1[n:],c1[n:])])'
    script=out/'rebuild_cavity_ledger.py'
    if token in code:code=replace_once(code,token,token[:-1]+',dtype=bool)')
    if 'float(dist.max()) if dist.size else None' not in code:code=replace_once(code,'float(dist.max())','float(dist.max()) if dist.size else None')
    script.write_text(code)
    subprocess.run([sys.executable,script,job,mesh],check=True)
    if sha(vol/'current-child.labels.csv')!=label_sha:raise ValueError('The proposed anatomical labels changed during allocation')
if __name__=='__main__':prepare(*(Path(s).resolve() for s in sys.argv[1:]))
