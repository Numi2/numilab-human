"""Integrate the partitioned gluteal material and its anatomical ports on the registered P1 carrier."""
import json,sys,time
from pathlib import Path
import manifold3d as m
import numpy as np
from preserve_muscle_cells import det
from condition_motion_weights import quality_denom
from pelvic_specimen import properties
FACE=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
from material_context import sha

def make_solid(x,f):
    identity=m.Manifold.reserve_ids(1)
    mesh=m.Mesh64(np.ascontiguousarray(x,dtype='f8'),np.ascontiguousarray(f,dtype='u8'),run_index=np.array([0,3*len(f)],'u8'),run_original_id=np.array([identity],'u4'),face_id=np.arange(len(f),dtype='u8'))
    geom=m.Manifold(mesh)
    if geom.status()!=m.Error.NoError:raise ValueError(geom.status())
    return geom,identity

def barycentric(points,p):
    z=(points-p[0])@np.linalg.inv(p[1:]-p[0]);return np.column_stack((1-z.sum(axis=1),z))

def integrate(capture,carrier,part,material,out):
    out.mkdir(exist_ok=True);report={};rho=properties(material)["density"]
    for sid in (29,30):
        start=time.monotonic();source=part/('muscle_%d.npz'%sid);d=np.load(source);v=d['vertices'];f=d['triangles'];g,identity=make_solid(v,f)
        base=capture/('entrance-%d.npz'%sid);x=np.load(base)['xyz'].astype(float)
        tp=carrier/str(sid)/'source.1.ele';t=np.loadtxt(tp,skiprows=1,dtype=int)[:,1:5];ref=det(x,t)/6
        if ref.min()<=0:raise ValueError('Inverted P1 carrier')
        consistent=np.zeros((len(t),4,4));volume=np.zeros(len(t));integral=np.zeros((len(t),4))
        boundary_cells=[];parent_ids=[];port_xyz=[];port_bary=[]
        for i,cell in enumerate(t):
            p=x[cell]
            tetra,_=make_solid(p,FACE);clip=tetra^g
            if clip.status()!=m.Error.NoError:raise ValueError((sid,i,clip.status()))
            if clip.is_empty():continue
            native=clip.to_mesh64();pts=np.array(native.vert_properties[:,:3]);ids=np.array(native.tri_verts,int)
            exact=clip.volume()
            if exact<0 or exact>ref[i]+max(2.e-18,ref[i]*1.e-9):raise ValueError((i,exact,ref[i]))
            if exact==0:continue
            # Signed polyhedron integration is exact for a linear basis, including nonconvex pieces.
            y=pts.mean(axis=0);tri=pts[ids];sub=(tri-y)
            weights=np.einsum('ij,ij->i',sub[:,0],np.cross(sub[:,1],sub[:,2]))/6
            volume[i]=weights.sum()
            b=barycentric(pts,p);bc=barycentric(y[None,:],p)[0]
            integral[i]=(weights[:,None]*(bc+b[ids].sum(axis=1))*.25).sum(axis=0)
            corner=np.concatenate((np.broadcast_to(bc,(len(ids),1,4)),b[ids]),axis=1)
            total=corner.sum(axis=1)
            consistent[i]=(np.einsum('n,ni,nj->ij',weights,total,total)+np.einsum('n,nki,nkj->ij',weights,corner,corner))/20
            if abs(volume[i]-exact)>max(2.e-17,1.e-7*exact):raise ValueError(('quadrature',i,volume[i],exact))
            run=np.repeat(native.run_original_id,np.diff(native.run_index)//3)
            owned=np.flatnonzero(run==identity);parent=np.array(native.face_id,dtype=int)[owned]
            sel=np.flatnonzero(d['parent_bank'][parent]!='muscle_'+str(sid))
            if len(sel):
                owned=owned[sel];boundary_cells.extend([i]*len(owned));parent_ids.extend(parent[sel]);port_xyz.extend(tri[owned]);port_bary.extend(b[ids[owned]])
        parent_ids=np.asarray(parent_ids,int);node_v=np.bincount(t.reshape(-1),weights=integral.reshape(-1),minlength=len(x))
        if node_v.min() < -1.e-16 or integral.min() < -1.e-16:raise ValueError(('negative mass',node_v.min(),integral.min()))
        np.testing.assert_allclose(consistent.sum(axis=2),integral,rtol=1e-8,atol=1e-22)
        np.testing.assert_allclose(volume,integral.sum(axis=1),rtol=1.e-9,atol=1.e-18)
        np.testing.assert_allclose(volume.sum(),g.volume(),rtol=2.e-7,atol=1.e-14)
        origin=v.mean(axis=0);faces=v[f]-origin;dv=np.einsum('ij,ij->i',faces[:,0],np.cross(faces[:,1],faces[:,2]))/6
        moment=(dv[:,None]*(origin+.25*faces.sum(axis=1))).sum(axis=0)
        np.testing.assert_allclose(node_v@x,moment,rtol=2.e-7,atol=1.e-14)
        ports=np.asarray(port_xyz);area=np.linalg.norm(np.cross(ports[:,1]-ports[:,0],ports[:,2]-ports[:,0]),axis=1)*.5
        original=v[f];whole_area=np.linalg.norm(np.cross(original[:,1]-original[:,0],original[:,2]-original[:,0]),axis=1)*.5
        a=np.bincount(parent_ids,weights=area,minlength=len(f));mask=d['parent_bank']!='muscle_'+str(sid)
        np.testing.assert_allclose(a[mask],whole_area[mask],rtol=2.e-5,atol=5.e-15)
        result=out/(str(sid)+'.npz')
        np.savez_compressed(result,vertices=x.astype('f4'),cells=t.astype('u4'),cell_volumes=volume,shape_integrals=integral,consistent_shape_integrals=consistent,port_owner_cell=np.asarray(boundary_cells,'u4'),port_partition_face=parent_ids.astype('u4'),port_vertices=ports,port_shape=np.asarray(port_bary))
        report[sid]=dict(source_sha256=sha(source),carrier_sha256=sha(tp),entrance_sha256=sha(base),output_sha256=sha(result),material_sha256=sha(material),carrier_cells=len(t),positive_material_cells=int((volume>0).sum()),partial_material_cells=int(((volume>1e-17)&(ref-volume>1e-17)).sum()),min_carrier_quality=(det(x,t)/quality_denom(x,t)).min(),volume_ml=float(volume.sum()*1e6),unallocated_muscle_mass_kg=rho*volume.sum(),negative_mass_rows=int((integral < -1e-16).sum()),minimum_shape_integral_m3=integral.min(),mass_first_moment_residual_kg_m=float(np.max(abs(node_v@x-moment))*rho),port_facets=len(ports),area_error_m2=float(max(abs(a[mask]-whole_area[mask]))),wall_s=time.monotonic()-start)
        print(sid,report[sid],flush=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':integrate(*map(Path,sys.argv[1:]))
