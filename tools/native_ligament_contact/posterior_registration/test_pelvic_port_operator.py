"""Check derivative, rigid invariance, and the full material moment tensor independently on both actual banks."""
import json,sys
from pathlib import Path
import numpy as np
from pelvic_specimen import properties
from material_context import sha

def test(part,projected,ports,material,out):
    rows={};spec=properties(material);rho=spec['density'];rng=np.random.default_rng(3948)
    for sid in (29,30):
        p=np.load(projected/(str(sid)+'.npz'));d=np.load(part/('muscle_%d.npz'%sid));b=np.load(ports/(str(sid)+'.npz'))
        v=d['vertices'];f=d['triangles'];x=p['vertices'].astype(float);t=p['cells'];mass=p['consistent_shape_integrals'];vol=p['cell_volumes']
        center=v.mean(0);tri=v[f]-center
        # Surface-only second moment, integrated over signed tets from the chosen origin.
        dv=np.einsum('fi,fi->f',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6;tv=tri.sum(1)
        surface=rho*(np.einsum('f,fi,fj->ij',dv,tv,tv)+np.einsum('f,fki,fkj->ij',dv,tri,tri))/20
        xt=x[t]-center;volume=rho*np.einsum('tij,tik,tjl->kl',mass,xt,xt)
        moment_error=float(np.max(abs(surface-volume)))
        np.testing.assert_allclose(volume,surface,rtol=1.e-10,atol=1.e-15)
        active=vol>0;t=t[active];vol=vol[active]
        inv=np.linalg.inv(x[t[:,1:]]-x[t[:,:1]])
        u=rng.normal(size=x.shape)*1.e-9
        direction=rng.normal(size=x.shape);direction/=np.linalg.norm(direction)
        def gradient(a):return (inv@(a[t[:,1:]]-a[t[:,:1]])).transpose(0,2,1)
        I=np.eye(3);F=I+gradient(u)
        assert np.linalg.det(F).min()>.9
        def energy(F):
            h=np.log(np.linalg.det(F))
            return vol@(.5*spec['mu']*((F**2).sum((1,2))-3-2*h)+.5*spec['lam']*h*h)
        def piola(F):
            d=spec['lam']*np.log(np.linalg.det(F))-spec['mu']
            return spec['mu']*F+d[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
        D=gradient(direction);h=2.e-9
        # Central virtual-work difference cancels equal isotropic reference terms.
        predicted=np.einsum('t,tij,tij',vol,piola(F),D)
        observed=(energy(F+h*D)-energy(F-h*D))/(2*h)
        rel=float(abs(predicted-observed)/max(abs(predicted),1.e-10))
        if rel>.003:raise ValueError(('Energy gradient',sid,predicted,observed,rel))
        # A rigid velocity field must give identical values on both sides during transfer.
        source_nodes=p['vertices'][p['cells'][p['port_owner_cell']]].astype(float)
        omega=np.array([-.91,.45,.72]);drift=np.array([.43,.24,-.36])
        def test_velocity(x):return np.cross(omega,x)+drift
        a=np.einsum('fki,fij->fkj',p['port_shape'],test_velocity(source_nodes))
        c=np.einsum('fki,fij->fkj',b['target_shape'],test_velocity(b['target_triangle'].astype(float)))
        e=float(np.max(abs(a-c)));assert e<1e-12
        rows[sid]=dict(independent_second_moment_error_kg_m2=moment_error,virtual_work_relative_difference=rel,rigid_velocity_jump_m_s=e,quadrature_sha256=sha(projected/(str(sid)+'.npz')))
    out.write_text(json.dumps(rows,indent=2)+'\n');print(rows)
if __name__=='__main__':test(*map(Path,sys.argv[1:]))
