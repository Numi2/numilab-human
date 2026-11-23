"""Build a conforming local two-material PLC at all measured bilateral cortical conflicts."""
from pathlib import Path
import json,sys
import manifold3d as md
import numpy as np
from build_carpal_cavity import boundary,shell,parts
from audit_registered_skeleton import digest


def run(mesh,witnesses,interfaces,broad,out):
    x=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3));tet=np.loadtxt(mesh/'mesh.ele',skiprows=1,dtype='i4')
    t=tet[:,1:5]-1;roles=tet[:,5]
    w=np.load(witnesses/'witnesses.npz');cross=np.load(interfaces/'crossed-facets.npz')
    reference=json.loads((broad/'result.json').read_bytes())
    if digest(mesh/'mesh.node')!=reference['source_material']:raise ValueError('Wrong physical material')
    out.mkdir()
    for side,ids in [('bilateral',(8,9,10,11,12,13))]:
        center=x[np.concatenate([w['ids'+str(b)] for b in ids])].mean(0);v=(x-center)*1000
        banks={};u=None
        for b in ids:
            y,inv=np.unique(w['cortex'+str(b)],axis=0,return_inverse=True);bf=inv[w['faces'+str(b)]]
            y=(y-center)*1000
            if np.einsum('ij,ij->',y[bf[:,0]],np.cross(y[bf[:,1]],y[bf[:,2]]))<0:bf=bf[:,[0,2,1]]
            k=md.Manifold.reserve_ids(1);banks[k]=b
            part=shell(y,bf,k);u=part if u is None else u+part
        touched=[w['ids'+str(b)] for b in ids]
        for b in ids:
            for mark in ('15O','15I','16I','16D'):
                k=str(b)+mark
                if k in cross:touched.append(cross[k].ravel())
        seed=np.unique(np.concatenate(touched))
        for ring in range(1,5):
            cells=np.flatnonzero(np.isin(t,seed).any(1));seed=np.unique(t[cells])
            # Expand a crop boundary only through existing material cells.
            # Two disconnected wedges touching at one native edge produce
            # four boundary uses despite each wedge being a valid solid.
            for attempt in range(32):
                extra=[]
                for mat in (15,16):
                    chosen=cells[roles[cells]==mat]
                    if not len(chosen):continue
                    f=boundary(t[chosen]);e=np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
                    stride=len(x);keys,counts=np.unique(e[:,0].astype('i8')*stride+e[:,1],return_counts=True)
                    bad=keys[counts!=2]
                    if not len(bad):continue
                    nodes=np.unique(np.r_[bad//stride,bad%stride]);near=np.flatnonzero((roles==mat)&(np.isin(t,nodes).sum(1)>=2))
                    edges=np.sort(t[near][:,[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]],axis=2)
                    found=np.isin(edges[:,:,0].astype('i8')*stride+edges[:,:,1],bad).any(1)
                    extra.extend(near[found])
                if not extra:break
                joined=np.union1d(cells,extra)
                if len(joined)==len(cells):raise ValueError('Original material edge has disjoint incident domains')
                cells=joined
            else:raise ValueError('Material edge closure did not finish')
            try:
                crop={};summary={}
                for mat in (15,16):
                    subset=cells[roles[cells]==mat];f=boundary(t[subset]);oid=md.Manifold.reserve_ids(1);s=shell(v,f,oid)
                    inside,rem=s.split(u);target=sum(reg[str(mat)]['removed_overlap_mm3'] for reg in reference['sides'].values())
                    if abs(inside.volume()-target)>max(1.e-6,abs(target)*1e-8):
                        raise ValueError(f'Incomplete {side} {mat} region: {inside.volume()}, expected {target}')
                    if rem.status()!=md.Error.NoError:raise ValueError('Invalid material solid')
                    crop[mat]=(f,oid,parts(rem));summary[mat]=dict(original_tets=len(subset),removed_overlap_mm3=inside.volume(),outside_mm3=rem.volume())
                break
            except ValueError as err:print('expand',side,ring,err,flush=True)
        else:raise ValueError('Local mesh cannot be closed')
        common={tuple(row) for row in np.sort(crop[15][0],axis=1)}&{tuple(row) for row in np.sort(crop[16][0],axis=1)}
        xyz=[];fs=[];markers=[];native_ids=[];src_faces=[];n=0
        for mat,(f,oid,(vc,tr,owner,face)) in crop.items():
            selected=[];mk=[]
            for i,(o,fi) in enumerate(zip(owner,face)):
                if o in banks:label=100+mat;region=banks[o]
                elif o==oid:
                    is_shared=tuple(sorted(f[fi])) in common
                    if is_shared and mat==16:continue
                    label=1516 if is_shared else mat;region=0
                else:raise ValueError('Missing source face provenance')
                selected.append(i);mk.append(label);native_ids.append(region);src_faces.append(int(fi))
            xyz.extend(vc);fs.extend(tr[selected]+n);markers.extend(mk);n+=len(vc)
        points=np.asarray(xyz)*.001+center
        y,inv=np.unique(points.astype('f4'),axis=0,return_inverse=True);faces=inv[np.array(fs)]
        area=np.linalg.norm(np.cross(y[faces[:,1]].astype(float)-y[faces[:,0]],y[faces[:,2]].astype(float)-y[faces[:,0]]),axis=1)
        pool=out/side;pool.mkdir();archive=pool/'cavity.npz'
        zero=np.flatnonzero(area==0)
        np.savez_compressed(archive,source_tetrahedra=cells,vertex_world_m=y,faces=faces,marker=markers,zero_faces=zero,
                            cortex_bank=native_ids,parent_face=src_faces,skin_parent_facets=crop[15][0],fat_parent_facets=crop[16][0])
        report=dict(source_material=reference['source_material'],ring=ring,materials=summary,cavity=digest(archive),center_m=center.tolist(),native_f32_degenerate_facets=len(zero))
        (pool/'cavity.json').write_text(json.dumps(report,indent=2)+'\n')
        print(side,'source cells',len(cells),'vertices',len(y),'facets',len(faces),'zero',len(zero),flush=True)


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
