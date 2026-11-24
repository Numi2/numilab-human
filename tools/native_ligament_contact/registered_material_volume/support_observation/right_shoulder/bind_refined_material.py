"""Replay the current scene's material basis and native mass carriers on the refined physical volume."""
import json
from pathlib import Path
import shutil,subprocess,sys
import numpy as np
from rebuild_tissue_source_mapping import link,write
from compile_connected_joint import digest


def run(w, v, out):
    r=json.loads((v/'result.json').read_bytes());patch=w/'material-import/volume';job=out/'source';dst=out/'carrier-replay'
    if not r['native']['native_imported'] or r['native']['density_outliers'] or r['native']['inverted']:raise ValueError('Missing native material verification')
    for k,h in r['pins'].items():
        if digest(v/k)!=h:raise ValueError('Changed volume '+k)
    base=w/'material-ancestry';prev=base/'carrier-replay';orig=w.parent/'conditioned'
    for p in [out/'witness-material',job/'deep-fat-native',job/'verified-connected-material',dst]:p.mkdir(parents=True,exist_ok=True)
    link(out/'witness-material/volume',v);link(out/'owner.locator',orig/'owner.locator');link(out/'current-p1',prev/'material-p1')
    link(job/'deep-fat-ledger',prev);link(job/'posed-app',prev/'posed-app');link(job/'verified-connected-material/scene',w/'native-cavity-bound')
    link(dst/'volume',v);link(dst/'material-p1',out/'witness-material/current-p1')
    for f in ('mesh.ele','cell-mass.f64'):link(job/'deep-fat-native'/f,patch/f)
    write(job/'deep-fat-native/result.json',dict(native_command=r['native_import_command']))
    overlay=np.load(v/'source-overlay.npz');src=overlay['from_cell'];target=overlay['to_cell']
    mass=np.fromfile(patch/'cell-mass.f64',dtype='<f8');n=len(np.fromfile(patch/'node-mass.f64',dtype='<f8'))
    mapping=np.arange(len(mass),dtype='i4');mapping[np.unique(src)]=-1
    previous=np.r_[np.arange(n,dtype='i4'),np.full(r['nodes']-n,-1,dtype='i4')]
    np.savez_compressed(v/'overlay.npz',old_to_new=mapping,source=src,target=target,fraction=overlay['mass_kg']/mass[src],previous_node_rows=previous)
    write(v/'result.json',r|dict(exit=0,inputs=r['pins']))
    material=np.loadtxt(patch/'mesh.ele',skiprows=1,usecols=5,dtype='i4')
    cavities=[]
    for role in (15,16):
        selected=material[src]==role
        cavities.append([role,np.unique(src[selected]).tolist(),np.unique(target[selected]).tolist(),0.,0.])
    write(v/'transfer.json',dict(cavities=cavities,cells=r['cells'],nodes=r['nodes'],elements=r['pins']['mesh.ele'],mass=r['pins']['cell-mass.f64']))
    owner=base/'rebuild_current_material_basis.py';shutil.copy2(owner,out/owner.name)
    p1done=dst/'material-p1/result.json'
    if not p1done.exists():
        with (out/'p1.log').open('w')as log:subprocess.run([sys.executable,out/owner.name,'--motion-witness'],stdout=log,stderr=subprocess.STDOUT,check=True)
    p1=json.loads((dst/'material-p1/result.json').read_bytes())
    print('P1 classifications',[(role,len(s['inadmissible_label_rows']))for role,s in p1['roles'].items()],flush=True)
    with (out/'carrier.log').open('w')as log:
        subprocess.run([sys.executable,orig/'rebuild_cavity_ledger.py',job,dst],stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=dict(owner=digest(owner),native_import=r['native'],p1=digest(dst/'material-p1/result.json'),mass_carriers=digest(dst/'native-replay/result.json'),flows=digest(dst/'flows/result.json'))
    write(out/'result.json',summary)

if __name__=='__main__':run(*(Path(a).resolve()for a in sys.argv[1:]))
