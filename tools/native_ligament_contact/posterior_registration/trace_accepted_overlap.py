"""Locate the deepest occupied material in an independently confirmed native cortical overlap."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from cortical_distance import CorticalSurface
from project_material_cells import make_solid,FACE
from prepare_material_interface import solid
from fit_contact_domain import points
from material_context import sha

def run(part,prev,assets,run,library,out):
    snapshot=run/'accepted-pelvic-state.csv';withrows=list(csv.DictReader(snapshot.open()))
    def vals(k):return {int(r['index']):np.array([float(r[a]) for a in 'xyzw']) for r in withrows if r['field']==k}
    pos=vals('relative_position');rot={b:Rotation.from_quat(q) for b,q in vals('orientation').items()}
    mat=vals('material_node');geom=np.stack([mat[i][:3] for i in range(len(mat))])
    manifest=json.loads((assets/'manifest.json').read_text());audit=json.loads((run/'solid-overlap.json').read_text())
    tri=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3].astype(float)
    banks=np.fromfile(assets/'cortical-samples.bin','f4').reshape(2,256,5,4)
    base=0;res={}
    for side,sid in enumerate((29,30)):
        p=np.load(part/'projected'/f'{sid}.npz');d=np.load(part/'partition-layers'/f'muscle_{sid}.npz')
        shape=make_solid(d['vertices'],d['triangles'])[0];v=p['vertices'].astype(float);t=p['cells']
        nodes,w,xyz,owner,bary=points(part,prev,sid);curr=geom[base:base+len(v)];base+=len(v)
        surface=np.einsum('ij,ijk->ik',w,curr[nodes]);near=cKDTree(surface)
        ports=np.load(part/'ports'/f'{sid}.npz')
        for key,cert in audit['muscles'][str(sid)]['cortex'].items():
            i=cert['worst_cell'];mask=[j for j,f in enumerate(manifest['sides'][side]['quadrature_indices']) if ports['target_bank'][f]=='bone_'+key]
            if i is None or not mask:continue
            row=banks[side,mask[0]].copy().view('u4');lo,n=row[4,:2];body=row[3,2]
            x=rot[128].inv().apply(rot[body].apply(tri[lo:lo+n].reshape(-1,3))+pos[body][:3]);faces=np.arange(len(x)).reshape(-1,3)
            bone=solid(x,faces,'cortex')[0];query=CorticalSurface(library,x,faces)
            ref=v[t[i]];current=curr[t[i]];cut=make_solid(ref,FACE)[0]^shape
            F=(np.linalg.inv(ref[1:]-ref[:1])@(current[1:]-current[:1])).T;shift=current[0]-F@ref[0]
            moved=cut.transform(np.column_stack((F,shift)));overlap=moved^bone;mesh=overlap.to_mesh64()
            a=np.asarray(mesh.vert_properties)[:,:3];ft=np.asarray(mesh.tri_verts)
            probes=np.vstack((a,a[ft].mean(1),a.mean(0)));gap=query.query(probes)[0];ix=int(gap.argmin())
            pts=probes[ix];b=(pts-current[0])@np.linalg.inv(current[1:]-current[:1]);weights=np.r_[1-b.sum(),b]
            owned=owner==i;polys=np.einsum('fai,fij->faj',bary[owned],curr[t[owner[owned]]])
            observed_samples=query.query(surface)[0]
            # Which deformed material boundary facets have vertices inside this individual intersection?
            frac=np.linspace(0,1,25);uv=np.array([[u,v,1-u-v] for u in frac for v in frac if u+v<=1])
            if len(polys):
                dense=np.einsum('ni,fij->fnj',uv,polys)
                gg=query.query(dense.reshape(-1,3))[0].reshape(len(polys),-1)
                fi,k=np.unravel_index(gg.argmin(),gg.shape)
                interior=dict(minimum_boundary_distance_m=float(gg[fi,k]),material_facet=int(np.flatnonzero(owned)[fi]),barycentric_on_facet=uv[k].tolist())
            else:interior=None
            res[f'{sid}/{key}']=dict(cell=i,solid_volume_ml=overlap.volume()*1e6,deepest_sampled_interior_distance_m=float(gap[ix]),material_barycentric=weights.tolist(),distance_to_nearest_registered_boundary_sample_m=near.query(pts)[0],minimum_current_boundary_sample_distance_m=float(observed_samples.min()),boundary_facets_on_cell=int(owned.sum()),interior=interior)
    out.write_text(json.dumps(dict(accepted_step=audit['accepted_step'],snapshot_sha256=sha(snapshot),witnesses=res),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
