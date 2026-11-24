"""Minimize the registered strain energy on connected high-distortion material patches."""
import argparse,ctypes,json,re
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from import_conditioned_interior import write_nodes
from supported_material_admission import geom

def run(source,attribution,pins_file,out,checker=None,checker_receipt=None,committed=None,coupled=False):
    volume=source/'volume'
    x,y=[np.loadtxt(volume/f,skiprows=1,usecols=(1,2,3)) for f in ('mesh.node','current.node')]
    src=y.copy()
    raw=np.loadtxt(volume/'mesh.ele',dtype=np.int32,skiprows=1,usecols=(1,2,3,4,5))
    all_t=raw[:,:4]-1
    peak=json.loads(attribution.read_bytes())['spans'][0]
    if peak['label']!='registered bonded skin and fat':raise ValueError('Unmatched witness')
    witness=peak['local']
    pins=json.loads(pins_file.read_bytes())['material_pins']
    mat={}
    for k,match in [(15,'skin_dermis'),(16,'adipose_low')]:
        f,h=next((Path(n),p) for n,p in pins.items() if match in n)
        if digest(f)!=h:raise ValueError('Wrong material')
        mat[k]=[1e6*float(re.search(r'parameter '+w+r'\s*:\s*MPa\s*=\s*([0-9.eE+-]+)',f.read_text())[1]) for w in ('shear','bulk')]
    flagged=[]
    for offset in range(0,len(raw),32768):
        t=all_t[offset:offset+32768]
        a=x[t[:,1:]]-x[t[:,:1]];b=y[t[:,1:]]-y[t[:,:1]]
        stretch=np.linalg.svd(np.linalg.solve(a,b),compute_uv=False)
        flagged.extend(np.flatnonzero((stretch[:,0]>2)|(stretch[:,-1]<.5))+offset)
    walls=np.load(volume/'material-walls.npz')
    code=(source/'binding/app/NumiHumanLoadedSkinPoseData.inc').read_text()
    anchors=np.array([int(s) for s in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',code,flags=re.M)])
    if not len(anchors):raise ValueError('Missing attachment pins')
    locked=np.unique(np.r_[anchors,24331,24538,26199,*[walls[k].ravel() for k in ('O','D','C')]])
    facets=np.load(source/'binding/contact/material-contact.npz')['facets'][:,5:8]
    if np.setdiff1d(facets,locked).size:raise ValueError('Unprotected cortical contact')
    # A patch includes the material on both sides of the shared interface.
    seed=np.unique(np.r_[all_t[flagged].ravel(),witness])
    first=np.isin(all_t,seed).any(axis=1)
    free=np.setdiff1d(np.unique(all_t[first]),locked)
    affected=np.flatnonzero(np.isin(all_t,free).any(axis=1))
    t=all_t[affected]; prop=np.array([mat[int(k)] for k in raw[affected,4]])
    mu,bulk=prop.T
    inv=np.linalg.inv(x[t[:,1:]]-x[t[:,:1]])
    grads=np.concatenate((-inv.sum(axis=2,keepdims=True),inv),axis=2).transpose(0,2,1)
    vr=np.linalg.det(x[t[:,1:]]-x[t[:,:1]])/6
    inc={int(n):[] for n in free}
    for i,row in enumerate(t):
        for j,n in enumerate(row):
            if n in inc:inc[n].append((i,j))
    stars={n:np.array(rows).T for n,rows in inc.items()}
    radii={}
    for n,(a,b) in stars.items():
        edges=np.linalg.norm(y[t[a]]-y[n],axis=2)
        radii[n]=.25*np.min(edges[edges>0])
    def local(a,positions):
        e=positions[:,1:]-positions[:,:1]
        return (inv[a]@e).transpose(0,2,1)
    def energy(F,a):
        lj=np.log(np.linalg.det(F))
        return np.sum(vr[a]*(.5*mu[a]*((F*F).sum((1,2))-3-2*lj)+.5*bulk[a]*lj**2))
    interval=None
    if checker is not None:
        if digest(checker)!=json.loads(checker_receipt.read_bytes())['checker']:raise ValueError('Wrong native checker')
        interval=ctypes.CDLL(str(checker)).scanCellIntervals
        fp=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
        interval.argtypes=[fp,fp,ctypes.c_size_t,ip,ctypes.c_size_t,fp]
    guards=[]
    if committed is not None:
        # Use the control's measured first-step displacement as a second geometry
        # sample. The real integration below remains responsible for its own motion.
        end=np.loadtxt(committed,skiprows=1,usecols=(1,2,3))
        if end.shape!=src.shape:raise ValueError('Unmatched motion sample')
        allrows=np.arange(len(t))
        for displacement in (np.zeros_like(src),end-src):
            points=(src+displacement)[t]
            F=local(allrows,points)
            J=np.linalg.det(F)
            log=np.log(J)
            W=.5*mu*((F*F).sum((1,2))-3-2*log)+.5*bulk*log**2
            _,q=geom(src+displacement,t)
            guards.append((displacement,np.minimum(.75,J),np.maximum(1.5,J),np.minimum(q,1.2e-5),np.maximum(W,.02*mu)))
    def safe_motion(n,a,points):
        simple=np.arange(4*len(a)).reshape(-1,4)
        for displacement,lo,hi,qmin,maximum_energy in guards:
            trial=points+displacement[t[a]]
            F=local(a,trial)
            J=np.linalg.det(F)
            if np.any((J<lo[a]-2e-9)|(J>hi[a]+2e-9)):return False
            _,q=geom(trial.reshape(-1,3),simple)
            if np.any(q<qmin[a]*(1-1e-9)):return False
            log=np.log(J)
            energy_density=.5*mu[a]*((F*F).sum((1,2))-3-2*log)+.5*bulk[a]*log**2
            if np.any(energy_density>maximum_energy[a]*(1+1e-8)+1e-6):return False
        return True
    def block(n):
        a,b=stars[n];g=grads[a,b]
        F=local(a,y[t[a]]);J=np.linalg.det(F);lj=np.log(J)
        fg=(F@g[:,:,None])[:,:,0]
        h=np.linalg.solve(F.transpose(0,2,1),g[:,:,None])[:,:,0]
        gradient=np.sum(vr[a,None]*(mu[a,None]*fg+(bulk[a]*lj-mu[a])[:,None]*h),axis=0)
        H=np.eye(3)*np.sum(vr[a]*mu[a]*(g*g).sum(1)) + (h.T*(vr[a]*(mu[a]+bulk[a]*(1-lj))))@h
        return gradient,H,energy(F,a)
    order=sorted(stars,key=lambda n:-np.linalg.norm(block(n)[0]))
    every=np.arange(len(t))
    def bounds():
        vv,q=geom(y,t)
        return dict(energy_J=float(energy(local(every,y[t]),every)),min_J=float(min(vv/vr)),max_J=float(max(vv/vr)),min_shape=float(min(q)),witness_force_N=float(np.linalg.norm(block(witness)[0])))
    if witness not in stars:raise ValueError('Terminal defect is a fixed boundary')
    baseline=bounds();progress=[]
    if coupled:
        from coupled_direction import fit
        if not guards or interval is None:raise ValueError('Coupled fit requires measured motion and native intervals')
        progress=fit(y,src,x,t,free,inv,vr,mu,bulk,guards,interval,radii,local,energy,geom,witness)
    for sweep in range(0 if coupled else 12):
        changed=0
        for n in order:
            a,b=stars[n];gradient,H,old=block(n)
            if np.linalg.norm(gradient)<1e-5:continue
            delta=-np.linalg.solve(H,gradient)
            points=y[t[a]].copy()
            for step in (1,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625):
                new=(y[n]+step*delta).astype('f4').astype(float)
                if np.linalg.norm(new-src[n])>radii[n] or np.array_equal(y[n],new):continue
                points[np.arange(len(a)),b]=new
                v,q=geom(points.reshape(-1,3),np.arange(4*len(a)).reshape(-1,4))
                J=v/vr[a]
                if np.min(J)<.4 or np.max(J)>2.5 or np.min(v)<=1e-18 or np.min(q)<1e-5:continue
                if energy(local(a,points),a)>=old or guards and not safe_motion(n,a,points):continue
                if interval is not None:
                    indices=np.arange(4*len(a),dtype=np.int32).reshape(-1,4);vals=np.empty((len(a),3))
                    if interval(np.ascontiguousarray(x[t[a]].reshape(-1,3)),np.ascontiguousarray(points.reshape(-1,3)),indices.size,indices,len(a),vals):raise ValueError('Failed native local interval')
                    margin=np.minimum(vals[:,0]-vals[:,1]-float(np.float32(.35)),2.5-vals[:,0]-vals[:,1])
                    if not np.all(margin>0):continue
                y[n]=new;changed+=1;break
        progress.append(dict(sweep=sweep+1,accepted_updates=changed,**bounds()))
        if not changed:break
    ids=np.flatnonzero((src!=y).any(axis=1))
    if np.intersect1d(ids,locked).size or not len(ids) or not progress[-1]['energy_J']<baseline['energy_J']:raise ValueError('No admissible energy reduction')
    write_nodes(volume/'current.node',out/'current.node',{n:y[n] for n in ids})
    result=dict(control_loaded=digest(volume/'current.node'),registered=digest(out/'current.node'),source_reference=digest(volume/'mesh.node'),
        source_mass=digest(volume/'node-mass.f64'),elements=digest(volume/'mesh.ele'),source_material=pins,coupled=coupled,motion_control=(digest(committed) if committed is not None else None),flagged_cells=len(flagged),free_nodes=len(free),affected_cells=len(t),changed_nodes=len(ids),
        max_displacement_m=float(np.linalg.norm(y-src,axis=1).max()),shifted_interface_nodes=int(np.isin(ids,walls['I']).sum()),baseline=baseline,progress=progress)
    (out/'fit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([baseline,progress[-1]]))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','attribution','pins_file','out'):p.add_argument(n,type=Path)
    p.add_argument('--coupled',action='store_true')
    p.add_argument('--committed',type=Path)
    p.add_argument('--checker',type=Path)
    p.add_argument('--checker-receipt',type=Path)
    run(**vars(p.parse_args()))
