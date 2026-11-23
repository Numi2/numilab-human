"""Cut both true material shells against the compiled cortex and retain native node and face ancestry."""
import json,sys
from pathlib import Path
import manifold3d as md
import numpy as np
from scipy.spatial import cKDTree
from audit_registered_skeleton import digest
from build_carpal_cavity import shell,parts


def canonical(x,f):
    v=np.ascontiguousarray(x[f],dtype='<f4')
    t=v.view('<u4').reshape(-1,3,3)
    order=np.lexsort((t[:,:,2],t[:,:,1],t[:,:,0]),axis=1)
    keys=np.ascontiguousarray(np.take_along_axis(t,order[:,:,None],axis=1)).reshape(-1,9).view('V36').ravel()
    return np.sort(keys)


def run(mesh,domains,conflicts,interfaces,out):
    x=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3))
    ref=json.loads((conflicts/'result.json').read_bytes())
    if digest(mesh/'mesh.node')!=ref['mesh']:raise ValueError('Material pose changed')
    w=np.load(conflicts/'witnesses.npz')
    tree=cKDTree(x);roles=w['roles'];out.mkdir()
    exact=json.loads((interfaces/'result.json').read_bytes())
    if exact['registered_bones']!=ref['bones']:raise ValueError('Wrong cortical witness')
    report={}
    for side,ids in [('l',(9,11,13)),('r',(8,10,12))]:
        z=np.load(domains/(side+'.npz'));center=z['pivot'].astype(float)
        banks={};bank_names={};cut=None
        for b in ids:
            points,remap=np.unique(w['cortex'+str(b)],axis=0,return_inverse=True)
            faces=remap[w['faces'+str(b)]];v=(points-center)*1000
            if np.einsum('ij,ij->',v[faces[:,0]],np.cross(v[faces[:,1]],v[faces[:,2]]))<0:faces=faces[:,[0,2,1]]
            owner=md.Manifold.reserve_ids(1);solid=shell(v,faces,owner)
            banks[b]=owner;bank_names[owner]=b
            cut=solid if cut is None else cut+solid
            if exact['interface_triangle_pairs'][str(b)].get('15O')!=0:
                raise ValueError('Outer skin needs reconstruction before this internal repair')
        bank_report={};retained={}
        for mat in (15,16):
            q=z['x'+str(mat)];faces=z['f'+str(mat)]
            err,indices=tree.query(q)
            if err.max()>1.e-12:raise ValueError('Shoulder domain does not map to current volume coordinates')
            owner=md.Manifold.reserve_ids(1);src=shell((q-center)*1000,faces,owner)
            inside,outside=src.split(cut)
            if any(s.status()!=md.Error.NoError for s in (cut,inside,outside)):raise ValueError('Cortex Boolean failed')
            vv,ff,labels,parent=parts(outside)
            surface=np.array([0 if i==owner else bank_names[int(i)] for i in labels],dtype='i4')
            # Parent facet IDs on cropped material boundaries resolve back to the
            # exact original global nodes. Cortex facets retain their bank and face.
            inherited=surface==0
            parent_nodes=np.full((len(ff),3),-1,'i4');parent_nodes[inherited]=indices[faces[parent[inherited]]]
            actual=(vv*.001+center).astype('<f4')
            lost_outer=None
            if mat==15:
                before=(roles[indices[faces]]=='O').all(1)
                after=np.zeros(len(ff),bool);after[inherited]=before[parent[inherited]]
                pre=canonical(q,faces[before]);post=canonical(actual,ff[after])
                lost_outer=len(np.setdiff1d(pre,post))
                if lost_outer or len(pre)!=len(post):
                    raise ValueError(f'{side} exterior skin changed: {len(pre)}, {len(post)}, {lost_outer}')
            emitted_ids,fi=np.unique(actual,axis=0,return_inverse=True);ef=fi[ff]
            collapsed=int(np.sum((ef[:,0]==ef[:,1])|(ef[:,1]==ef[:,2])|(ef[:,2]==ef[:,0])))
            residual=float((outside^cut).volume())
            delta=float(src.volume()-inside.volume()-outside.volume())
            if abs(delta)>max(1e-7,src.volume()*1e-10) or abs(residual)>1e-6:raise ValueError('Unresolved continuum overlap or material volume balance')
            retained.update({f'x{mat}':vv,f'f{mat}':ff,f'bank{mat}':surface,f'parent_face{mat}':parent,f'source_nodes{mat}':parent_nodes})
            bank_report[mat]=dict(control_mm3=float(src.volume()),removed_overlap_mm3=float(inside.volume()),remaining_mm3=float(outside.volume()),
                overlap_after_mm3=residual,volume_ledger_error_mm3=delta,source_outer_facets=int(before.sum()) if mat==15 else None,
                changed_outer_facets=lost_outer,cut_facets={str(b):int((surface==b).sum()) for b in ids},collapsed_float32_faces=collapsed)
        retained['center_m']=center
        np.savez_compressed(out/(side+'.npz'),**retained)
        report[side]=bank_report
    (out/'result.json').write_text(json.dumps(dict(source_material=ref['mesh'],cortical_registration=ref['bones'],
        layer_domains={s:digest(domains/(s+'.npz')) for s in ('l','r')},sides=report),indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
