"""Regenerate authoritative P1 attachment and mass-carrier support for appended material samples."""
from pathlib import Path
import json,sys,subprocess
import numpy as np
from compile_connected_joint import digest


def link(dst,src):
    if not dst.exists():dst.symlink_to(src.resolve())
    if dst.resolve()!=src.resolve():raise ValueError('Mismatched source link')


def write(f,data):f.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def rebuild(j,compiled,out):
    out.mkdir(exist_ok=True);v=compiled/'volume';r=json.loads((compiled/'result.json').read_bytes())
    if r['exit'] or r['native']['density_outliers'] or abs(r['mass_change_kg'])>1e-10:raise ValueError('Unverified physical mass')
    orig=j/'conditioned';base=orig/'witness-material/volume'
    if digest(base/'cell-mass.f64')!=r['input_mass']:raise ValueError('Foreign parent material')
    old_n=len(np.fromfile(base/'node-mass.f64','<f8'));old_c=len(np.fromfile(base/'cell-mass.f64','<f8'))
    z=np.load(v/'source-overlay.npz');source=z['from_cell'];target=z['to_cell'];ids=z['old_cells']
    mapping=np.arange(old_c,dtype='i4');mapping[ids]=-1
    nd=np.r_[np.arange(old_n,dtype='i4'),np.full(r['native']['nodes']-old_n,-1,dtype='i4')]
    np.savez_compressed(v/'overlay.npz',old_to_new=mapping,source=source,target=target,fraction=z['fraction'],previous_node_rows=nd)
    write(v/'result.json',dict(exit=0,native=r['native'],inputs=r['pins']))
    cav=[[int(r0),c['source'],c['destination_rows'],c['row_error'],c['column_error']] for c in r['admitted'] for r0 in [np.loadtxt(base/'mesh.ele',skiprows=1,usecols=5,dtype='i4')[c['source'][0]]]]
    write(v/'transfer.json',dict(cavities=cav,cells=r['native']['cells'],nodes=r['native']['nodes'],elements=r['pins']['mesh.ele'],mass=r['pins']['cell-mass.f64']))
    (out/'witness-material').mkdir(exist_ok=True)
    link(out/'witness-material/volume',v);link(out/'owner.locator',orig/'owner.locator');link(out/'current-p1',orig/'carrier-replay/material-p1')
    old=orig/'rebuild_current_material_basis.py';src=old.read_text()
    key="        nd=np.arange(len(x),dtype='i4')"
    if src.count(key)!=1:raise ValueError('Unknown P1 compiler')
    src=src.replace(key,"""        overlay=np.load(v/'overlay.npz')
        nd=overlay['previous_node_rows'] if 'previous_node_rows' in overlay else np.arange(len(x),dtype='i4')
        if nd.shape!=(len(x),) or (nd< -1).any():raise ValueError('Invalid node ancestry')""")
    # Reuse the original integration owner. Appended nodes have no old classification.
    src=src.replace('        # No new nodes; the owner detects changed Float32 positions by comparison.','')
    p=out/'rebuild_current_material_basis.py';p.write_text(src)
    with (out/'p1.log').open('w')as log:subprocess.run([sys.executable,p,'--motion-witness'],stdout=log,stderr=subprocess.STDOUT,check=True)
    p1=out/'witness-material/current-p1';pinfo=json.loads((p1/'result.json').read_bytes())
    if any(k['inadmissible_label_rows']for k in pinfo['roles'].values()):raise ValueError('Unresolved material label ancestry')
    job=out/'source';dst=out/'carrier-replay'
    for folder in [job/'deep-fat-native',job/'verified-connected-material',dst]:folder.mkdir(parents=True,exist_ok=True)
    link(job/'deep-fat-ledger',orig/'carrier-replay');link(job/'posed-app',orig/'carrier-replay/posed-app');link(job/'verified-connected-material/scene',j/'posterior-interface-082/native-pair')
    for n in ('mesh.ele','cell-mass.f64'):link(job/'deep-fat-native'/n,base/n)
    write(job/'deep-fat-native/result.json',dict(native_command=r['native_import_command']))
    link(dst/'volume',v);link(dst/'material-p1',p1)
    with (out/'carrier.log').open('w')as log:
        result=subprocess.run([sys.executable,orig/'rebuild_cavity_ledger.py',job,dst],stdout=log,stderr=subprocess.STDOUT)
    report=dict(source=digest(old),generated=digest(p),basis=sha_or_none(p1/'result.json'),roles=pinfo['roles'],carrier_exit=result.returncode,
        replay=sha_or_none(dst/'native-replay/result.json'),flow=sha_or_none(dst/'flows/result.json'))
    write(out/'result.json',report);print(report,flush=True)
    if result.returncode:raise ValueError('Native carrier validation did not pass')


def sha_or_none(p):return digest(p) if p.is_file() else None
if __name__=='__main__':rebuild(*(Path(s).resolve()for s in sys.argv[1:]))
