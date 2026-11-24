"""Measure registration deformation and its material forces on the exact observed node."""
import argparse,json,re
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from certify_interior_motion import response

def run(volume,material_pins,star,native,result):
    tetfile=volume/'mesh.ele'
    t=np.loadtxt(tetfile,skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32)
    ref,load,end=[np.loadtxt(p,skiprows=1,usecols=(1,2,3)) for p in (volume/'mesh.node',volume/'current.node',native/'physical-accepted.node')]
    site=json.loads(star.read_bytes()); node=int(site['source']['joint_node_id'])-1
    sources=json.loads(material_pins.read_bytes())['material_pins']
    r={}
    for region,match in [(15,'skin_dermis'),(16,'adipose_low')]:
        material,sha=next((Path(path),h) for path,h in sources.items() if match in Path(path).name)
        if digest(material)!=sha:raise ValueError('Wrong constitutive source')
        text=material.read_text()
        def get(k):
            return float(re.search(r'parameter '+k+r'\s*:\s*MPa\s*=\s*([0-9.eE+-]+)',text)[1])*1e6
        mu,bulk=get('shear'),get('bulk')
        selected=(t[:,4]==region); ids=np.flatnonzero(selected);elements=t[ids,:4]-1
        one=elements[(elements==node).any(axis=1)]
        vertices,local=np.unique(one,return_inverse=True);local=local.reshape(one.shape)
        i=np.flatnonzero(vertices==node).item()
        role={}
        for name,x in [('registered',load),('first_accepted',end)]:
            stretch=[]; compression=[];volume0=[]; witness={}
            for off in range(0,len(ids),32768):
                rows=elements[off:off+32768]
                d0=ref[rows[:,1:]]-ref[rows[:,:1]]
                d=x[rows[:,1:]]-x[rows[:,:1]]
                f=np.linalg.solve(d0,d)
                sv=np.linalg.svd(f,compute_uv=False)
                stretch.extend(sv[:,0]);compression.extend(sv[:,-1]);volume0.extend(np.linalg.det(d0)/6)
                k=int(np.argmax(sv[:,0]))
                if sv[k,0]>witness.get('maximum_principal_stretch',0):
                    witness=dict(zero_based_cell=int(ids[off+k]),maximum_principal_stretch=float(sv[k,0]),minimum_principal_stretch=float(sv[k,-1]),J=float(np.prod(sv[k])))
            a,b,vol=map(np.asarray,(stretch,compression,volume0))
            ee,ff,_=response(ref[vertices],x[vertices],local,mu,bulk)
            role[name]=dict(worst=witness,principal_stretch_count_quantiles=np.quantile(a,[.5,.9,.99,.999]).tolist(),
                count_stretch_gt_2=int((a>2).sum()),count_stretch_gt_10=int((a>10).sum()),source_volume_fraction_stretch_gt_2=float(vol[a>2].sum()/vol.sum()),
                count_compression_lt_half=int((b<.5).sum()),cells=len(a),source_mu_pa=mu,source_logJ_coefficient_pa=bulk,
                witness_material_internal_force_N=ff[i].tolist(),witness_star_strain_energy_J=ee)
        r[region]=role
    result.write_text(json.dumps(dict(materials=r,source_star=digest(star),source_reference=digest(volume/'mesh.node'),registered=digest(volume/'current.node'),accepted=digest(native/'physical-accepted.node'),scope='Reference-material deformation and constitutive forces evaluated on captured coordinates. Dynamic inertia, contact and external traction are excluded.'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('volume','material_pins','star','native','result'):p.add_argument(n,type=Path)
    run(**vars(p.parse_args()))
