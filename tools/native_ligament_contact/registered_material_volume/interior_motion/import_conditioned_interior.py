"""Carry material mass by overlap through admitted interior relocations and call the native importer."""
import argparse,ctypes,json,subprocess
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from supported_material_admission import geom
from transfer_cavities import intersection_fraction

def write_nodes(src,dst,updates):
    with src.open('rb') as i,dst.open('wb') as o:
        o.write(i.readline())
        for n,line in enumerate(i):
            if n in updates:
                if int(line.split(maxsplit=1)[0])!=n+1:raise ValueError('Unordered node')
                line=(str(n+1)+' '+' '.join(format(v,'.17g') for v in updates[n])+'\n').encode()
            o.write(line)
def run(parent,proposal,certificate,out):
    src=parent/'witness-material/volume'
    p=json.loads(proposal.read_bytes());c=json.loads(certificate.read_bytes())
    if c['proposal_sha256']!=digest(proposal) or digest(src/'mesh.ele')!=p['physical_mesh']:raise ValueError('Stale physical input')
    out.mkdir(exist_ok=False)
    admitted={r['vertex'] for r in c['sites'] if r['passed']}
    sites=[s for s in p['selected'] if s['vertex'] in admitted]
    if not sites:raise ValueError('No qualified sites')
    x,y=[np.loadtxt(src/n,skiprows=1,usecols=(1,2,3)) for n in ('mesh.node','current.node')]
    t=np.loadtxt(src/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype=np.int32)-1
    role=np.loadtxt(src/'mesh.ele',skiprows=1,usecols=5,dtype=np.int32)
    wall=np.load(src/'material-walls.npz'); locked=np.unique(np.concatenate([a.ravel() for a in wall.values()]))
    if np.intersect1d(list(admitted),locked).size:raise ValueError('Material boundary moved')
    dm=np.fromfile(src/'cell-mass.f64',dtype='<f8'); masses=dm.copy()
    X=x.copy();Y=y.copy();records=[];frows=[];trows=[];parts=[]
    for r in sites:
        i=r['vertex'];rows=np.array(r['affected_cells']);n=np.array(r['neighbor_nodes']);w=np.array(r['local_weights'])
        if set(rows)!=set(np.flatnonzero((t==i).any(axis=1))):raise ValueError('Incomplete material star')
        X[i]+=w@(x[n]-x[i]);Y[i]+=w@(y[n]-y[i])
        X[i]=X[i].astype(np.float32);Y[i]=Y[i].astype(np.float32)
        old=t[rows];new=old.copy();new[new==i]=len(x)
        verts=np.vstack((x,X[i]))
        V=geom(x,old)[0];Vnew=geom(verts,new)[0]
        frac=np.array([[intersection_fraction(a,b,verts) for b in new] for a in old])
        row_err=float(abs(frac.sum(axis=1)-1).max());col_err=float(abs((frac.T@V)/Vnew-1).max())
        if max(row_err,col_err)>3e-9:raise ValueError(f'Unresolved source overlap at vertex {i}: {row_err} {col_err}')
        frac/=frac.sum(axis=1)[:,None];masses[rows]=frac.T@dm[rows]
        src_ix,dest_ix=np.nonzero(frac)
        frows.extend(rows[src_ix]);trows.extend(rows[dest_ix]);parts.extend(frac[src_ix,dest_ix])
        records.append(dict(vertex=i,role=int(role[rows[0]]),cells=len(rows),source_fraction_error=row_err,destination_volume_relative_error=col_err,
                            source_mass_kg=float(dm[rows].sum()),new_mass_kg=float(masses[rows].sum()),
                            density_min=float(min(masses[rows]/Vnew)),density_max=float(max(masses[rows]/Vnew))))
    for name in ('mesh.ele','material-walls.npz','current-child.labels.csv'):
        subprocess.run(['/bin/cp','-cp',src/name,out/name],check=True)
    for name,array in [('mesh.node',X),('current.node',Y)]:
        write_nodes(src/name,out/name,{i:array[i] for i in admitted})
    masses.tofile(out/'cell-mass.f64')
    nodes=sum(np.bincount(t[:,j],weights=masses*.25,minlength=len(X)) for j in range(4))
    if min(nodes)<=0:raise ValueError('Massless material point')
    nodes.astype('<f8').tofile(out/'node-mass.f64')
    np.savez_compressed(out/'source-overlay.npz',from_cell=frows,to_cell=trows,fraction=parts,mass_kg=dm[frows]*np.array(parts),old_cells=np.unique(frows),new_cells=np.unique(trows))
    # Certify the identical reference-to-registered admission interval used by the parent.
    j=parent.parent.parent.parent
    chk=Path((j/'storage/interval-checker-locator.txt').read_text().split()[1])
    if digest(chk)!=json.loads((j/'verified-connected-material/interval-audit.json').read_bytes())['checker']:raise ValueError('Interval checker changed')
    dp=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    fn=ctypes.CDLL(str(chk)).scanCellIntervals
    fn.argtypes=[dp,dp,ctypes.c_size_t,ip,ctypes.c_size_t,dp];vals=np.empty((len(t),3))
    if fn(np.ascontiguousarray(X),np.ascontiguousarray(Y),len(X),np.ascontiguousarray(t),len(t),vals):raise ValueError('Interval checker failed')
    margin=np.minimum(vals[:,0]-vals[:,1]-float(np.float32(.35)),2.5-vals[:,0]-vals[:,1])
    if min(margin)<=0:raise ValueError('Material interval failed')
    cmd=json.loads((parent.parent/'remaining-fold/collar-2/fold-volume/native-launch.json').read_bytes())['command']
    cmd[1]=str(out.resolve())
    with (out/'native-import.json').open('w') as so,(out/'native-import.stderr').open('w') as se:
        subprocess.run(cmd,stdout=so,stderr=se,check=True)
    receipt=json.loads((out/'native-import.json').read_bytes())
    if not receipt['native_imported'] or any(receipt[k]!=0 for k in ('inverted','below_shape','below_volume','nonfinite','density_outliers','zero_mass_nodes')):
        raise ValueError('Native material admission failed')
    # Replay the held-out physical samples with the same interpolation.
    defects={}
    for name,origin in [('matched',parent/'accepted-material-07'),('held_out',parent/'half-step-material-08')]:
        u=np.loadtxt(origin/'physical-accepted.node',skiprows=1,usecols=(1,2,3));updated=u.copy()
        for r in sites:
            i=r['vertex'];updated[i]+=np.array(r['local_weights'])@(u[r['neighbor_nodes']]-u[i])
        old,new={},{}
        for m in (15,16):
            for arr,record in [(u,old),(updated,new)]:
                v,s=geom(arr,t[role==m]);record[str(m)]=dict(inverted=int((v<=0).sum()),low_quality=int((s<1e-5).sum()),low_volume=int((v<=1e-18).sum()))
        defects[name]=dict(previous=old,conditioned=new)
    result=dict(exit=0,native=receipt,inputs={k:digest(out/k) for k in ('mesh.node','current.node','mesh.ele','cell-mass.f64','node-mass.f64','current-child.labels.csv','source-overlay.npz','material-walls.npz')},source_mesh=p['physical_mesh'],native_command=cmd,
        updated_sites=records,minimum_interval_margin=float(min(margin)),captured_replay=defects,
        scope='Native material import and replay of captured coordinates with conserved source mass. New coupled native advancement is a distinct check.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('parent','proposal','certificate','out'):p.add_argument(a,type=Path)
    run(**vars(p.parse_args()))
