"""Install the local subdivision in the existing registered TetGen, wall and mass inputs."""
import json
from pathlib import Path
import subprocess, sys
import numpy as np
from compile_connected_joint import digest
from compile_tissue_repairs import save_nodes
from refine_shoulder_material import split_rows
from map_shoulder_mass import map_mass

def read(p):
    return json.loads(p.read_bytes())


def compile_patch(w, patch, out):
    base=w/'material-import/volume'; r=read(patch/'result.json'); z=np.load(patch/'material.npz')
    if digest(patch/'material.npz')!=r['material'] or digest(base/'mesh.ele')!=r['elements'] or digest(base/'material-walls.npz')!=r['walls'] or digest(base/'cell-mass.f64')!=r['mass_source']:
        raise ValueError('Changed parent registration')
    sources=[base/'mesh.node',base/'current.node']
    if [digest(p) for p in sources]!=r['input'][:2]:
        raise ValueError('Changed reference or loaded pose')
    out.mkdir(exist_ok=True,parents=True)
    ids=z['ids']; rows=z['source_rows']; cells=z['t']; parents=z['source_parent']; role=z['roles']
    for k,p in enumerate(sources):
        xyz=np.loadtxt(p,skiprows=1,usecols=(1,2,3))
        if not np.array_equal(xyz[ids],z['X'][k,:len(ids)]):
            raise ValueError('Patch boundary is stale')
        n=len(xyz)
        save_nodes(out/p.name,np.vstack([xyz,z['X'][k,len(ids):]]))
    node=np.r_[ids,np.arange(n,n+len(z['X'][0])-len(ids))]
    raw=np.loadtxt(base/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    if not np.array_equal(raw[rows,:4]-1,ids[z['base']]):
        raise ValueError('Wrong parent topology')
    # Keep an original cell index within every parent and append its other children.
    first=np.unique(parents,return_index=True)[1]
    dest=np.empty(len(cells),dtype=int);dest[first]=rows
    more=np.ones(len(cells),dtype=bool);more[first]=False
    dest[more]=np.arange(len(raw),len(raw)+more.sum())
    def extend(x):
        return np.concatenate((x,np.empty((more.sum(),)+x.shape[1:],dtype=x.dtype)))
    elements=extend(raw);elements[dest,:4]=node[cells]+1;elements[dest,4]=role
    m=extend(np.fromfile(base/'cell-mass.f64',dtype='<f8'));m[dest]=z['mass']
    tags=np.loadtxt(base/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    if not np.array_equal(tags[:,0],np.arange(len(tags))) or not np.array_equal(tags[:,1],raw[:,4]):
        raise ValueError('Invalid material parent labels')
    current_mass,packet_source,packet_target,packet_mass,mapping=map_mass(z,np.fromfile(base/'cell-mass.f64',dtype='<f8')[rows],tags[rows,1:])
    m[dest]=current_mass
    labels=extend(tags[:,1:]);labels[dest]=tags[rows[parents],1:]
    for k in (15,16):
        if abs(m[elements[:,4]==k].sum()-np.fromfile(base/'cell-mass.f64',dtype='<f8')[raw[:,4]==k].sum())>1e-11:
            raise ValueError('Regional mass drift')
    with (out/'mesh.ele').open('w')as f:
        f.write(f'{len(elements)} 4 1\n')
        for q in np.array_split(np.arange(len(elements)),48):
            np.savetxt(f,np.column_stack((q+1,elements[q])),fmt='%d')
    with (out/'current-child.labels.csv').open('w')as f:
        with (base/'current-child.labels.csv').open()as source:f.write(source.readline())
        for q in np.array_split(np.arange(len(labels)),48):
            np.savetxt(f,np.column_stack((q,labels[q])),fmt='%d',delimiter=',')
    m.tofile(out/'cell-mass.f64')
    nodal=sum(np.bincount(elements[:,i]-1,weights=m*.25,minlength=int(node.max())+1) for i in range(4))
    if nodal.min()<=0 or abs(nodal.sum()-m.sum())>1e-11:raise ValueError('Invalid nodal mass or domain')
    nodal.astype('<f8').tofile(out/'node-mass.f64')
    original_walls=np.load(base/'material-walls.npz');walls={};wall_ancestry={}
    for k in original_walls.files:
        original=original_walls[k];initial=z['rows_'+k]; faces=z['original_'+k];parents0=initial.copy()
        for a,b,mid in z['splits']:
            faces,mask=split_rows(faces,a,b,mid)
            parents0=np.r_[parents0[~mask],parents0[mask],parents0[mask]]
        if not np.array_equal(faces,z[k]):raise ValueError('Lost wall-face ancestry')
        keep=np.ones(len(original),dtype=bool);keep[initial]=False
        walls[k]=np.concatenate((original[keep],node[faces]))
        wall_ancestry[k]=np.r_[np.flatnonzero(keep),parents0]
    np.savez_compressed(out/'material-walls.npz',**walls)
    np.savez_compressed(out/'source-overlay.npz',from_cell=rows[packet_source],to_cell=dest[packet_target],mass_kg=packet_mass,geometric_parent=rows[parents],geometric_child=dest,geometric_fraction=z['fraction'],node_splits=node[z['splits']],**{'parent_wall_'+k:v for k,v in wall_ancestry.items()})
    cmd=read(w/'material-import/result.json')['native_import_command'].copy();cmd[1]=str(out.resolve())
    with (out/'native-import.json').open('w') as log,(out/'native-import.stderr').open('w')as err:
        result=subprocess.run(cmd,stdout=log,stderr=err)
    if result.returncode:raise ValueError('Native anatomical volume import failed')
    observed=read(out/'native-import.json')
    if not observed['native_imported'] or any(observed[k]for k in ('below_shape','below_volume','inverted','nonfinite','density_outliers','zero_mass_nodes')):raise ValueError('Native geometry or mass gate failed')
    summary=dict(mass_mapping=mapping,input=r['material'],nodes=len(nodal),cells=len(elements),native_import_command=cmd,native=observed,pins={p.name:digest(p)for p in out.iterdir()if p.name!='result.json'})
    (out/'result.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(observed,flush=True)

if __name__=='__main__':compile_patch(*(Path(a).resolve()for a in sys.argv[1:]))
