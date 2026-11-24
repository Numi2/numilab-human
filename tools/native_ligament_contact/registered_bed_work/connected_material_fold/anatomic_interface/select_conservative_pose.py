"""Safeguard the two regional fits at emitted precision, then select by measured bone penetration."""
from pathlib import Path
import ctypes,sys,json
import numpy as np
from supported_material_admission import geom
from compile_connected_joint import digest
from audit_registered_skeleton import closed_solid
from partition_carpal_material import solid
from build_carpal_cavity import boundary
from rebind_ligament_material import banks
from compare_supported_postures import poses
from scipy.spatial.transform import Rotation
def run(work,lib):
    q=work.parent;j=q.parent.parent;v=q/'qualified-material'
    x=np.loadtxt(v/'mesh.node',skiprows=1,usecols=(1,2,3));y=np.loadtxt(v/'current.node',skiprows=1,usecols=(1,2,3));el=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=el[:,:4]-1
    G=np.load(v/'material-walls.npz');f=np.ascontiguousarray(np.vstack([G[k]for k in 'OIDC']),'u4');z=y.copy();receipts=[]
    checker=Path((j/'storage/interval-checker-locator.txt').read_text().split()[1]);fn=ctypes.CDLL(str(checker)).scanCellIntervals;dp=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS');fn.argtypes=[dp,dp,ctypes.c_size_t,ip,ctypes.c_size_t,dp]
    arr=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS');hit=ctypes.CDLL(str(lib)).batch;hit.argtypes=[arr,arr,arr,ctypes.c_uint,arr];hit.restype=None
    proof=json.loads((v/'bone-overlap/result.json').read_bytes())['results'];B=banks(Path(json.loads((q/'gravity-scene/launch.json').read_bytes())['argv'][3]));P=poses(j/'distal-registered-native/target-pose.json')
    for role,bn in [('r_forearm',[14,16]),('pelvis',[18,19])]:
        source=np.load(work/(role+'-pose.npz'));refined=np.load(work/'locally-refined'/(role+'-pose.npz'));N=np.union1d(source['nodes'],refined['nodes']);a=y.copy();b=y.copy();a[source['nodes']]=source['positions'];b[refined['nodes']]=refined['positions']
        # Both regions have disjoint changed-node sets; protect the base elsewhere.
        ts=np.flatnonzero(np.isin(t,N).any(1));T=np.ascontiguousarray(t[ts]);fs=np.flatnonzero(np.isin(f,N).any(1));rest,_=geom(x,T);L=np.minimum.reduce([y,a,b]);H=np.maximum.reduce([y,a,b]);flo=L[f].min(1);fhi=H[f].max(1)
        C=np.flatnonzero(((flo<=fhi[fs].max(0))&(fhi>=flo[fs].min(0))).all(1));cand=[]
        for i in fs:
            ix=C[((flo[C]<=fhi[i])&(fhi[C]>=flo[i])).all(1)&(C!=i)]
            if len(ix):cand.extend(zip(np.minimum(ix,i),np.maximum(ix,i)))
        pairs=np.unique(np.array(cand,'u4').reshape(-1,2),axis=0)
        parts={}
        for k in bn:
            (body,_),(tr,pts,_)=next(p for p in B.items()if p[0][1]==k);verts=(Rotation.from_quat(P[body,3:]).apply(pts)+P[body,:3]).astype('f4').astype('f8');org=verts.mean(0);_,_,rig=closed_solid((verts-org)*1000,tr);near=np.ones(len(t),bool)
            for axis in range(3):near&=(L[:,axis][t].min(1)<=verts[:,axis].max())&(H[:,axis][t].max(1)>=verts[:,axis].min())
            parts[k]=(org,rig,[boundary(t[near&(el[:,4]==mat)])for mat in [15,16]])
        trials=[];best=None
        for frac in (0.,.125,.25,.375,.5,.625,.75,.875,1.):
            zz=z.copy();zz[N]=(a[N]+frac*(b[N]-a[N])).astype('f4');vol,shape=geom(zz,T);ratio=vol/rest;vals=np.empty((len(T),3));fn(x,zz,len(x),T,len(T),vals)
            invalid=int((~((ratio>=.4)&(ratio<=2.5)&(shape>=1e-5)&(vol>1e-18)&(vals[:,0]-vals[:,1]>float(np.float32(.35)))&(vals[:,0]+vals[:,1]<2.5))).sum())
            h=np.zeros(len(pairs),'u1');hit(zz,f,pairs,len(pairs),h);r={'fraction':frac,'invalid':invalid,'wall_crossings':int(h.sum())};trials.append(r)
            if invalid or h.any():continue
            er={}
            for k,(org,rig,grps)in parts.items():
                er[k]={}
                for label,tt in zip(['skin','fat'],grps):
                    n,i=np.unique(tt,return_inverse=True);sol=solid((zz[n]-org)*1000,i.reshape(-1,3));over=sol^rig
                    if over.status()!=rig.status():raise ValueError('Solid boolean failed')
                    er[k][label]=max(0.,over.volume())-proof[str(k)][label]
            r['changed_mm3']=er;score=sum(sum(pair.values())for pair in er.values())
            if score<0 and (best is None or score<best[0]):best=(score,zz,r)
        if best:z=best[1]
        receipts.append(dict(region=role,selected=best[2]if best else None,trials=trials))
    out=work/'selected-pose';out.mkdir(exist_ok=True)
    n=np.flatnonzero(np.any(z!=x,axis=1));field=out/'material-transfer.npz';np.savez_compressed(field,nodes=n,positions=z[n].astype('f4'))
    with (out/'current.node').open('w') as stream:
        stream.write(f'{len(z)} 3 0 0\n');np.savetxt(stream,np.c_[np.arange(1,len(z)+1),z],fmt=['%d','%.17g','%.17g','%.17g'])
    rec=dict(regions=receipts,field=digest(field),nodes=digest(out/'current.node'),initial=digest(v/'current.node'))
    (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n');print([(r['region'],r['selected'])for r in receipts])
if __name__=='__main__':run(Path(__file__).resolve().parent,Path(sys.argv[1]))
