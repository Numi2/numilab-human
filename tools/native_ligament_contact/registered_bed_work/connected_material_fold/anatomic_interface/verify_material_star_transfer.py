"""Verify the covered material's mass moments, reference deformation and interfaces."""
from pathlib import Path
import json
import numpy as np
from compile_connected_joint import digest
from material_cavity_flips import boundary
from supported_material_admission import geom

def run(work):
    r=work/'witness-material';vol=r/'volume';base=work.parent/'qualified-material';read=lambda f:json.loads(f.read_bytes())
    m=read(vol/'result.json');p=read(r/'current-p1/result.json');cr=read(r/'recorded-mass/native-replay/result.json')
    if m['exit'] or cr['exit'] or any(role['inadmissible_label_rows']for role in p['roles'].values()):raise ValueError('Unqualified native material')
    z=np.load(vol/'source-overlay.npz');source=z['old_cells'];dest=z['new_cells']
    xyz=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    x=xyz(vol/'mesh.node');y=xyz(vol/'current.node')
    if not np.array_equal(x[:-1],xyz(base/'mesh.node')) or not np.array_equal(y[:-1],xyz(work/'selected-pose/current.node')):raise ValueError('Existing material geometry moved')
    t0=np.loadtxt(base/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1;t=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    untouched=np.setdiff1d(np.arange(len(t0)),source)
    if not np.array_equal(t[untouched],t0[untouched]) or boundary(t0[source])!=boundary(t[dest]):raise ValueError('Cavity exterior changed')
    oldmass=np.fromfile(base/'cell-mass.f64','<f8');newmass=np.fromfile(vol/'cell-mass.f64','<f8')
    if not np.array_equal(oldmass[untouched],newmass[untouched]):raise ValueError('Mass changed outside cavity')
    def moments(cells,weights):
        pts=y[cells];q=np.einsum('n,nj->j',weights,pts.mean(1));M=weights.sum()
        # Inertia of the actual lumped FEM nodal representation, measured at the old cavity center.
        rel=pts-anchor
        second=np.einsum('n,nki,nkj->ij',weights*.25,rel,rel)
        I=np.eye(3)*np.trace(second)-second
        return q/M,I
    anchor=np.average(y[t0[source]].mean(1),axis=0,weights=oldmass[source])
    a,A=moments(t0[source],oldmass[source]);b,B=moments(t[dest],newmass[dest])
    def strain(c):
        DX=x[c[:,1:]]-x[c[:,:1]];dy=y[c[:,1:]]-y[c[:,:1]]
        F=np.linalg.solve(DX,dy).transpose(0,2,1)
        return (F.transpose(0,2,1)@F-np.eye(3))*.5
    a0=strain(t0[source]);a1=strain(t[dest])
    m0=oldmass[source].sum();m1=newmass[dest].sum()
    record=dict(source_cells=len(source),new_cells=len(dest),unchanged_cells=len(untouched),unchanged_vertices=len(x)-1,existing_coordinates_bitwise_equal=True,external_oriented_faces=len(boundary(t[dest])),same_wall_hash=digest(vol/'material-walls.npz')==digest(base/'material-walls.npz'),mass_before_kg=m0,mass_after_kg=m1,change_in_material_centroid_m=np.linalg.norm(b-a),change_in_gravity_torque_norm_Nm=9.81*m1*np.linalg.norm((a-b)[:2]),relative_local_lumped_inertia_change=np.linalg.norm(B-A)/np.linalg.norm(A),whole_body_relative_centroid_shift_m=np.linalg.norm(b-a)*m1/97.131950283934117,maximum_reference_pose_green_strain_before=float(np.max(np.abs(a0))),maximum_reference_pose_green_strain_after=float(np.max(np.abs(a1))),active_cell_carriers=read(r/'recorded-mass/packets/result.json')['trace'])
    v0=geom(x,t0[source])[0];v1=geom(x,t[dest])[0]
    delta=np.average(a1,axis=0,weights=v1)-np.average(a0,axis=0,weights=v0)
    record['cavity_volume_mean_green_strain_change_frobenius']=float(np.linalg.norm(delta))
    record['cavity_volume_mean_green_strain_change_fraction']=float(np.linalg.norm(delta)/max(np.linalg.norm(np.average(a0,axis=0,weights=v0)),1e-15))
    r0=np.sqrt(np.average(np.square(a0).sum((1,2)),weights=v0)); r1=np.sqrt(np.average(np.square(a1).sum((1,2)),weights=v1))
    record['cavity_rms_green_strain_before']=float(r0)
    record['cavity_rms_green_strain_after']=float(r1)
    if not record['same_wall_hash'] or abs(m1-m0)>1e-17:raise ValueError('Changed material boundary or mass')
    path=r/'transfer-check.json';path.write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':run(Path(__file__).resolve().parent)
