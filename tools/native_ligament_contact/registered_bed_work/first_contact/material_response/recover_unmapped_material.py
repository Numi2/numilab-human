"""Recover material ancestry, or retriangulate an unoccupied free fat facet when no source fat existed."""
import json,sys
from pathlib import Path
import numpy as np
from build_carpal_cavity import FACES
from fit_carpal_clearance import geom
from partition_carpal_material import solid
from verify_carpal_mesh import classify
from audit_registered_skeleton import digest
from retained_material_ancestry import run as trace

def run(mesh,overlap,plc,bones,out,paths):
    d=json.loads((overlap/'result.json').read_bytes());z=np.load(paths[-1]);t=z['t'].copy();x=z['x'].astype(float);r=z['role'];f=z['facets'];mk=z['markers'];out.mkdir()
    original=np.load(paths[0])['x'].astype(float);src=np.load(plc)['source_tetrahedra'];raw=np.loadtxt(mesh/'mesh.ele',skiprows=1,dtype='i4')[src];coord=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3));points=coord[raw[:,1:5]-1];lo=points.min(1);hi=points.max(1)
    missing=d['uncovered_children'];remove=[];keep=[];report=[]
    for j in missing:
        p=original[t[j]];center=p.mean(0);s=solid((p-center)*1000,FACES)
        vol=0.
        for k in np.flatnonzero((raw[:,5]==r[j])&((lo<=p.max(0))&(hi>=p.min(0))).all(1)):
            inter=s^solid((points[k]-center)*1000,FACES)
            if inter.status()!=s.status():raise ValueError('Unknown parent intersection')
            vol+=inter.volume()
        if vol>0:keep.append(j);continue
        if r[j]!=16 or z['locked'][t[j]].any():raise ValueError('An unowned non-fat or fixed cell requires new authoring')
        faces=t[j][FACES];surface=[];internal=[]
        for row in faces:
            key=np.sort(row);ix=np.flatnonzero((np.sort(f,axis=1)==key).all(1));labels=set(mk[ix]);neigh=np.flatnonzero(np.isin(t,row).sum(1)==3)
            if len(neigh)==1 and labels=={16}:surface.append((row,ix))
            elif len(neigh)==2 and (not labels or labels=={1000}) and np.all(r[neigh]==16):internal.append((row,ix))
            else:raise ValueError('Unmapped cell touches a preserved interface')
        if len(surface)!=2 or len(internal)!=2:raise ValueError('Free surface does not admit a two-triangle diagonal switch')
        delete=np.concatenate([idx for _,idx in surface+internal]);f=np.delete(f,delete,axis=0);mk=np.delete(mk,delete)
        f=np.vstack((f,*[row[::-1] for row,_ in internal]));mk=np.r_[mk,16,16];remove.append(j)
        report.append(dict(old_child=j,initial_source_overlap_mm3=vol,initial_tet_mm3=float(s.volume()),emitted_tet_mm3=float(geom(x[t[[j]]])[0][0]*1e9),vertices=t[j].tolist(),source_material='free deep surface'))
    kept=np.setdiff1d(np.arange(len(t)),remove);role,_=classify(x,t[kept],f[mk!=1000],mk[mk!=1000])
    if not np.array_equal(role,r[kept]) or not np.array_equal(np.unique(t),np.unique(t[kept])):raise ValueError('Material connectivity or a source vertex was lost')
    np.savez_compressed(out/'domain.npz',**{k:z[k] for k in z.files if k not in ('facets','markers','role','t','outer')},t=t[kept],role=role,facets=f,markers=mk,outer=np.sort(f[np.isin(mk,[15,16,115,116])],axis=1))
    rec=dict(original=digest(paths[-1]),domain=digest(out/'domain.npz'),original_spatial_map=digest(overlap/'overlap.npz'),removed=report,remaining_material_tracks=keep)
    if keep:
        target=out/'trace-input';target.mkdir();(target/'overlap.npz').hardlink_to((overlap/'overlap.npz').resolve());dd={**d,'uncovered_children':keep};(target/'result.json').write_text(json.dumps(dd)+'\n')
        trace(mesh,target,plc,paths,out/'advected')
        rec['advected']=json.loads((out/'advected/result.json').read_bytes())['material_tetrahedra']
    (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n')
if __name__=='__main__':
    mesh,overlap,plc,bones,out,*paths=map(Path,sys.argv[1:]);run(mesh,overlap,plc,bones,out,paths)
