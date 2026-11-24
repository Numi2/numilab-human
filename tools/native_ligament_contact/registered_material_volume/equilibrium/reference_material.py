"""Test a source-coordinate repair against unchanged per-cell mass and the native importer."""
import csv,ctypes,json,re,subprocess,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from import_conditioned_interior import write_nodes
from reference_fit import separation_fit
from supported_material_admission import geom

def run(source,control,probe,basefit,predicate,out):
    v=source/'volume';out.mkdir()
    read=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    orig=read(v/'current.node');X=read(v/'mesh.node');Y=read(basefit/'current.node')
    poses=[Y.copy(),read(probe/'physical-accepted.node'),Y+(read(control/'physical-accepted.node')-orig)]
    w=np.load(v/'material-walls.npz')
    pairs=json.loads((probe/'wall-crossings.json').read_text())['witnesses']['2-2']
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32);all_t=raw[:,:4]-1
    wallnodes=np.unique(w['D'][np.unique(pairs)])
    nodes=np.unique(all_t[np.isin(all_t,wallnodes).any(1)])
    anchors=[int(s) for s in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',(source/'binding/app/NumiHumanLoadedSkinPoseData.inc').read_text(),re.M)]
    fixed=np.unique(np.r_[anchors,w['O'].ravel(),w['I'].ravel(),w['C'].ravel(),np.load(source/'binding/contact/material-contact.npz')['facets'][:,5:8].ravel()])
    nodes=np.setdiff1d(nodes,fixed);rows=np.flatnonzero(np.isin(all_t,nodes).any(1));t=all_t[rows]
    if np.any(X[nodes]!=Y[nodes]) or np.any(raw[rows,4]!=16):raise ValueError('Source is not the isolated unloaded fat region')
    masses=np.fromfile(v/'cell-mass.f64','<f8')[rows];v0=geom(X,t)[0];qmin=[np.minimum(1e-5,geom(p,t)[1]) for p in poses]
    faces=np.ascontiguousarray(np.concatenate([w[k] for k in 'OIDC']),dtype='u4');touch=np.flatnonzero(np.isin(faces,nodes).any(1));near=set()
    for p in poses:
        lo=p[faces].min(1);hi=p[faces].max(1)
        for i in touch:
            neigh=np.flatnonzero(((lo<=hi[i]+4e-6)&(hi>=lo[i]-4e-6)).all(1))
            near.update(tuple(sorted((i,j))) for j in neigh if i!=j)
    ij=np.ascontiguousarray(sorted(near),dtype='u4');hit=np.empty(len(ij),'u1')
    arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS');batch=ctypes.CDLL(str(predicate)).batch;batch.argtypes=[arg,arg,arg,ctypes.c_uint,arg];batch.restype=None
    def check(shift):
        prior=X[nodes].copy();X[nodes]=prior+shift;vr,qref=geom(X,t);X[nodes]=prior
        rho=masses/vr
        if np.any((vr<=1e-18)|(qref<1e-5)|(rho<850)|(rho>950)):return False
        for i,p in enumerate(poses):
            prior=p[nodes].copy();p[nodes]=prior+shift;vol,q=geom(p,t);batch(p,faces,ij,len(ij),hit);p[nodes]=prior
            if np.any(q<qmin[i]-1.e-14) or np.any(vol/vr<.4 if i==0 else vol/vr<.35) or np.any(vol/vr>2.5) or np.any(hit):return False
        return True
    new,hist=separation_fit(t,X,poses,w['D'],pairs,nodes,check,geom,masses)
    (out/'conditioning.json').write_text(json.dumps(hist,indent=2)+'\n')
    if new is None:raise ValueError('No source registration passed the mass, density and motion constraints')
    delta=new-X[nodes];changed=nodes[np.any(delta!=0,1)]
    X[nodes]=new;Y[nodes]+=delta
    volume=out/'volume';volume.mkdir()
    for name,points,parent in [('mesh.node',X,v),('current.node',Y,basefit)]:
        write_nodes(parent/name,volume/name,{n:points[n] for n in changed})
    for name in ('mesh.ele','material-walls.npz','cell-mass.f64','node-mass.f64','current-child.labels.csv'):
        (volume/name).symlink_to((v/name).resolve())
    cmd=json.loads((v/'result.json').read_text())['native_command'];cmd[1]=str(volume)
    with (volume/'native-import.json').open('w') as f,(volume/'import.log').open('w') as err:subprocess.run(cmd,stdout=f,stderr=err,check=True)
    nr=json.loads((volume/'native-import.json').read_text());assert nr['native_imported'] and not any(nr[k] for k in ('inverted','below_shape','density_outliers','zero_mass_nodes','below_volume','nonfinite'))
    rho=masses/geom(X,t)[0]
    receipt=dict(reference=digest(volume/'mesh.node'),parent_reference=digest(v/'mesh.node'),loaded=digest(volume/'current.node'),material_mass=digest(volume/'cell-mass.f64'),node_mass=digest(volume/'node-mass.f64'),topology=digest(volume/'mesh.ele'),modified_nodes=changed.tolist(),updated_nodes=len(changed),local_cells=len(rows),affected_cell_mass_kg=float(masses.sum()),min_density=float(min(rho)),max_density=float(max(rho)),maximum_source_shift_m=float(np.linalg.norm(delta,axis=1).max()),patch_mass_first_moment_change_kg_m=(np.fromfile(v/'node-mass.f64','<f8')[nodes]@delta).tolist(),importer_result=nr,scope='Reference and loaded registration with immutable mass and connectivity. Live motion requires the regenerated registration and P1 bind.')
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
if __name__=='__main__':run(*(Path(n).resolve() for n in sys.argv[1:]))
