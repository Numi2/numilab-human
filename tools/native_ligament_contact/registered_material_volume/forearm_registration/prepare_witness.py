"""Compose consecutive local cavity transfers with the immediately preceding material state."""
from pathlib import Path
import json,sys,subprocess
from build_bed_audit import sha
def link(dst,src):
    src=src.resolve(strict=True)
    if not dst.exists(): dst.symlink_to(src)
    if dst.resolve()!=src:raise ValueError('Different owner for '+str(dst))
def run(prior,scene,out):
    h=out.parent;v=out/'volume';old=prior/'admitted-transfer';p=prior/'admitted-prep';work=h/'prep';s=work/'witness-material';s.mkdir(parents=True,exist_ok=True)
    link(work/'owner.locator',p/'owner.locator');link(work/'current-p1',old/'material-p1');link(s/'volume',v)
    rec=json.loads((out/'result.json').read_bytes())
    required=['mesh.node','mesh.ele','current.node','cell-mass.f64','node-mass.f64','material-walls.npz','current-child.labels.csv']
    (v/'result.json').write_text(json.dumps(dict(exit=0,native=rec['native'],inputs={k:sha(v/k) for k in required}),indent=2)+'\n')
    # The existing P1 compiler resolves its input directory against its own source file.
    script=work/'rebuild_current_material_basis.py';script.write_bytes((p/script.name).read_bytes())
    subprocess.run([sys.executable,script,'--motion-witness'],check=True)
    link(out/'material-p1',s/'current-p1');basis=json.loads((out/'material-p1/result.json').read_bytes())
    if any(role['inadmissible_label_rows'] for role in basis['roles'].values()):raise ValueError('Inadmissible anatomical source cells')
    job=work/'allocator';(job/'verified-connected-material').mkdir(parents=True,exist_ok=True)
    link(job/'verified-connected-material/scene',scene);link(job/'deep-fat-ledger',old);link(job/'posed-app',old/'posed-app')
    oldv=job/'deep-fat-native';oldv.mkdir(exist_ok=True)
    for name in ('cell-mass.f64','mesh.ele'):link(oldv/name,old/'volume'/name)
    (oldv/'result.json').write_text(json.dumps(dict(native_command=rec['native_command'])))
    label=sha(v/'current-child.labels.csv')
    subprocess.run([sys.executable,p/'rebuild_cavity_ledger.py',job,out],check=True)
    if sha(v/'current-child.labels.csv')!=label:raise ValueError('Source identity changed during allocation')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
