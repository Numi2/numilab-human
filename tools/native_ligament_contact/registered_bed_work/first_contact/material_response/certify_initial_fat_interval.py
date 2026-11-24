"""Correct the initial bonded interface using the same native determinant interval as the compiler."""
from pathlib import Path
import ctypes,sys,json,re
import numpy as np
from compile_connected_joint import digest
from certify_transported_material import measure,run as unused
from layer_contacts import run as audit_layers
def run(job,native,scene,lib,layerlib,out):
    g=np.loadtxt(native/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    x,y=[np.ascontiguousarray(np.loadtxt(native/s,skiprows=1,usecols=(1,2,3))) for s in ('mesh.node','current.node')]
    arg=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    fn=ctypes.CDLL(str(lib)).scanCellIntervals;fn.argtypes=[arg,arg,ctypes.c_size_t,ip,ctypes.c_size_t,arg]
    def intervals(q,t):
        t=np.ascontiguousarray(t,'i4');v=np.empty((len(t),3));assert fn(x,q,len(x),t,len(t),v)==0;return v
    def margin(v):return np.minimum(v[:,0]-v[:,1]-float(np.float32(.35)),2.5-v[:,0]-v[:,1])
    old=intervals(y,g);fail=np.flatnonzero(margin(old)<=0);log=(scene/'stderr.log').read_text();w=re.findall(r'cell=(\d+) material=(\d+) center=(\S+) radius=(\S+)',log)
    assert len(fail)==len(w)==1 and int(w[0][0])==fail[0] and np.allclose(old[fail,:2],np.array(w[0][2:],float),rtol=2e-9)
    cell=int(fail[0]);v=g[cell];C=np.flatnonzero(np.isin(g,v).any(1));lt=g[C];ref,_=measure(x,lt);b=y[v].copy();z=np.load(job/'registered-boundary/registered-boundary.npz')
    anchors=json.loads((job.parent/'ligament-equilibrium-070/scene-registration/registration.json').read_bytes());pinned=anchors['first_contact_new']+[t[2] for t in anchors['anchors']]
    assert np.all(np.isin(v,z['I_node_joint'])) and not np.isin(v,np.r_[pinned,z['O_node_joint'],np.load(job/'deep-fat-cortex/registered-cortical-facets.npz')['cell_side_material_bank_nodes'][:,-3:].ravel()]).any()
    t=b[1:]-b[:1];grad=np.array([np.cross(t[1],t[2]),np.cross(t[2],t[0]),np.cross(t[0],t[1])]);grad=np.vstack([-grad.sum(0),grad]);d0,_=measure(x,g[cell:cell+1]);grad/=d0[0]
    mass=np.fromfile(native/'node-mass.f64','<f8')[v];options=[]
    for mask in range(1,16):
        active=np.array([(mask>>i)&1 for i in range(4)],bool)
        if active.sum()<2:continue
        # Minimize the mass-weighted squared change for a prescribed first-order
        # volume correction, with zero change in the patch center of mass.
        direction=np.zeros((4,3));d=grad[active]/mass[active,None];d-=np.average(d,axis=0,weights=mass[active]);direction[active]=-d
        denom=-np.sum(grad*direction)
        if denom<=0:continue
        direction/=denom
        for goal in np.arange(.065,.162,.003):
            candidate=(b+goal*direction).astype('f4').astype(float);y[v]=candidate
            evals=intervals(y,lt);marg=margin(evals);d,l=measure(y,lt);J=d/ref
            if np.any(marg<=0) or marg[np.searchsorted(C,cell)]<.002 or np.any(d<=6e-18) or np.any(d<l*1e-5) or min(J)<.4 or max(J)>2.5:continue
            dist=np.linalg.norm(candidate-b,axis=1);moment=(candidate-b).T@mass
            options.append((float(dist.max()),float(np.linalg.norm(moment)),mask,goal,candidate.copy(),float(marg.min())))
            break
    y[v]=b;out.mkdir()
    if not options:raise ValueError('No connected center-of-mass preserving pose adjustment was admitted')
    maximum,roundoff,mask,goal,chosen,worst=min(options,key=lambda a:a[:2]);y[v]=chosen
    new=intervals(y,g);marg=margin(new);d,l=measure(y,g);rest,_=measure(x,g);J=d/rest
    if np.any(marg<=0) or not np.isfinite(new).all() or np.any(d<=6e-18) or np.any(d<l*1e-5) or min(J)<.4 or max(J)>2.5:raise ValueError('Whole-volume failure')
    ids=v[np.any(chosen!=b,axis=1)];p=out/'current.node';rows={int(i):y[i] for i in ids}
    with (native/'current.node').open() as f,p.open('x') as outp:
        outp.write(next(f))
        for i,line in enumerate(f):
            if i in rows:line=str(i+1)+' '+' '.join(format(float(a),'.17g') for a in rows[i])+'\n'
            outp.write(line)
    groups=[z[k+'_node_joint'][z[k+'_face_nodes']] for k in 'OID'];audit_layers(y,groups,out/'initial-layers.json',layerlib);layers=json.loads((out/'initial-layers.json').read_bytes())
    result=dict(reference=digest(native/'mesh.node'),original=digest(native/'current.node'),updated=digest(p),compiler_witness=cell,original_interval=old[cell].tolist(),updated_interval=new[cell].tolist(),minimum_whole_volume_margin=float(min(marg)),worst_changed_margin=worst,certified_cells=len(g),moved_nodes=ids.tolist(),max_displacement_um=maximum*1e6,rounding_first_moment_error_kg_m=roundoff,loaded_jacobian_range=[float(J.min()),float(J.max())],exact_layers=layers['pair_counts'],interval_checker=digest(lib),source_scene=digest(scene/'launch.json'))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if np.any(layers['pair_counts']):raise ValueError('Connected layer crossing')
    print(result,flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
