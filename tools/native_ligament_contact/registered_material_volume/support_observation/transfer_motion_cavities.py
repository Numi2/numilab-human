"""Apply accepted cavity geometry to the existing mass-overlay and native importer pipeline."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
from transfer_cavities import run as transfer
from build_bed_audit import sha

def run(v,cavities,checker,importer,out):
    out.mkdir();r=json.loads(cavities.read_bytes());raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1
    job=out/'inputs';folder=job/'interval-certified-scene';folder.mkdir(parents=True);(folder/'material').symlink_to(v)
    (job/'verified-connected-material').symlink_to(v);(job/'storage').mkdir();(job/'storage/interval-checker-locator.txt').symlink_to(checker)
    fit=out/'fitted';fit.mkdir();(fit/'result.json').write_text(json.dumps(dict(reference=r['source_coordinates'],loaded=r['loaded_coordinates'],elements=r['topology'])))
    ancestry=[dict(material=p['label'][0],old=t[p['old']].tolist(),new=p['new'])for p in r['patches']];(fit/'ancestry.json').write_text(json.dumps(ancestry))
    src=[np.array([i for p in r['patches'] if p['label'][0]==k for i in p['old']],dtype=int) for k in (15,16)]
    np.savez(fit/'patches.npz',source0=src[0],source1=src[1]);volume=out/'volume';transfer(job,fit,volume)
    overlay=np.load(volume/'overlay.npz');ids=overlay['old_to_new'];old=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8');count=int(np.max(overlay['target']))+1
    labels=np.full((count,4),-1,dtype='i8');rows=np.flatnonzero(ids>=0);labels[ids[rows]]=old[rows,1:]
    for a,b in zip(overlay['source'],overlay['target']):
        if labels[b,0]>=0 and not np.array_equal(labels[b],old[a,1:]):raise ValueError('Anatomical ancestry conflict')
        labels[b]=old[a,1:]
    if np.any(labels[:,0]<0):raise ValueError('Missing source label')
    with (volume/'current-child.labels.csv').open('w')as g,(v/'current-child.labels.csv').open()as f:
        g.write(f.readline())
        for lo in range(0,count,32768):
            hi=min(count,lo+32768);np.savetxt(g,np.column_stack([np.arange(lo,hi),labels[lo:hi]]),fmt='%d',delimiter=',')
    (volume/'material-walls.npz').symlink_to(v/'material-walls.npz')
    cmd=json.loads(importer.read_bytes())['native_command'].copy();cmd[1]=str(volume)
    with (volume/'native-import.json').open('w')as f,(volume/'import.log').open('w')as e:subprocess.run(cmd,stdout=f,stderr=e,check=True)
    status=json.loads((volume/'native-import.json').read_bytes());totals=json.loads((v/'native-import.json').read_bytes())
    for k in ('inverted','below_shape','below_volume','nonfinite','density_outliers','zero_mass_nodes'):
        if status[k]:raise ValueError(k)
    for k in ('skin_kg','fat_kg'):
        if abs(status[k]-totals[k])>1e-10:raise ValueError('Tissue mass changed')
    print({k:status[k] for k in ('nodes','cells','skin_kg','fat_kg','maximum_nodal_error_kg')});(out/'result.json').write_text(json.dumps(dict(native=status,source=sha(cavities),mass=sha(volume/'cell-mass.f64'),node_mass=sha(volume/'node-mass.f64'),native_command=cmd),indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
