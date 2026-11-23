"""Resolve actual cortical edges and check shortest-path reactions and solid clearance in three dimensions."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import minimize,nnls
from scipy.spatial.transform import Rotation
from audit_fiber_load_bounds import read_pose
from rebind_ligament_material import banks
from compile_connected_joint import digest
from audit_registered_skeleton import closed_solid


class Cortex:
    def __init__(self,body,sid,rec,pose):
        f,v,raw=rec;self.body=body;self.sid=sid
        self.rotation=Rotation.from_quat(pose[3:]);self.center=np.array(pose[:3])*1000.
        # Vertex seams in the detailed bone must share the same physical edge.
        self.local,self.first,key=np.unique(v,axis=0,return_index=True,return_inverse=True)
        self.xyz=self.rotation.apply(self.local)*1000+self.center
        f=key[f];self.faces=f
        pts=self.xyz[f];self.normal=np.cross(pts[:,1]-pts[:,0],pts[:,2]-pts[:,0])
        sign=np.einsum('ij,ij->',self.normal,pts[:,0]-self.xyz.mean(0));self.normal*=np.sign(sign)
        self.normal/=np.maximum(1.e-25,np.linalg.norm(self.normal,axis=1))[:,None]
        e=np.sort(np.stack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]),axis=1),axis=2)
        self.edges,self.inverse=np.unique(e.reshape(-1,2),axis=0,return_inverse=True)
        self.verts=self.xyz[self.edges];self.dd=self.verts[:,1]-self.verts[:,0];self.sq=np.einsum('ij,ij->i',self.dd,self.dd)
        self.lo=self.xyz.min(0);self.hi=self.xyz.max(0);self.solid=None
        counts=np.bincount(self.inverse)
        if np.all(counts==2):
            self.solid=closed_solid(self.xyz,f)[2]
        self.tris=self.xyz[f]
        self.face_lo=self.tris.min(1);self.face_hi=self.tris.max(1)

    def feature(self,p):
        t=np.clip(np.einsum('ij,ij->i',p-self.verts[:,0],self.dd)/np.maximum(self.sq,1e-20),0,1)
        e=self.verts[:,0]+t[:,None]*self.dd
        dist=np.linalg.norm(p-e,axis=1);i=dist.argmin()
        return float(dist[i]),int(i),float(t[i])


def line_length(y):
    seg=np.diff(y,axis=0);L=np.linalg.norm(seg,axis=1);unit=seg/np.maximum(L[:,None],1e-12)
    return float(L.sum()),unit[:-1]-unit[1:]


def verify(y,owners):
    # Intersections with an oriented closed cortex partition the segment.
    # Membership of each resulting open interval is independently tested with a nonparallel exterior ray.
    ray=np.array([.3614544,.6520908,.666823])
    worst=[];tol=1e-5 # millimetres, duplicate endpoint classification
    for j,(a,b) in enumerate(zip(y,y[1:])):
        length=np.linalg.norm(b-a)
        for o in owners:
            if np.any(np.minimum(a,b)>o.hi+tol) or np.any(np.maximum(a,b)<o.lo-tol):continue
            if o.solid is None:
                # For open banks directly audit every potentially intersected cortical triangle.
                from shoulder_inputs import old
                ids=np.flatnonzero(np.all((o.face_lo<=np.maximum(a,b)+tol)&(o.face_hi>=np.minimum(a,b)-tol),axis=1))
                if len(ids):
                    verts=o.tris[ids].reshape(-1,3)*.001
                    hits=old.audit.lp.vector_hits(a*.001,b*.001,(verts,np.arange(len(verts)).reshape(-1,3)))
                    if hits:worst.append([j,o.sid,'open surface crossing'])
                continue
            hits=o.solid.ray_cast(a,b)
            u=sorted({0.,1.,*(np.clip(float(np.dot(np.asarray(k.position)-a,b-a)/max(length**2,1e-30)),0.,1.) for k in hits)})
            for lo,hi in zip(u,u[1:]):
                if length*(hi-lo)<tol:continue
                p=a+(b-a)*(lo+hi)*.5
                # Count only strict interior; midpoints on cortical facets are intended contact.
                start=p-tol*ray;end=p+6000.*ray
                hs=o.solid.ray_cast(start,end)
                strict=[k for k in hs if np.linalg.norm(np.asarray(k.position)-p)>2*tol]
                if len(strict)==len(hs) and len(strict)%2:worst.append([j,o.sid,float(length*(hi-lo))])
    return worst


def run(parent,source,sections,output):
    table=json.loads((source/'source-routes.json').read_text())
    donor=next((source/'bone-candidate').glob('*.nhbones'))
    if digest(donor)!=table['bone_sha256']:raise ValueError('Unexpected authoring donor')
    original=banks(donor);bone=next((parent/'supported-bones').glob('*.nhbones'));dest=banks(bone)
    captured=parent/'native-ligament-material/stderr.log'
    pose=read_pose(captured)
    for owner in [k for k in dest if k[1] in (8,9,10,11,12,13)]:
        if not all(np.array_equal(original[owner][i],dest[owner][i]) for i in (0,2)):
            raise ValueError(f'Triangle identity or source coordinates changed for {owner}')
    library={};output.mkdir(exist_ok=True)
    for fi in sorted(sections.glob('*.json')):
        row=json.loads(fi.read_text());ids=row['broadphase_bones']
        if row['pose']!=digest(captured) or row['bones']!=digest(bone):raise ValueError('Mixed native geometries')
        for (b,s),rec in dest.items():
            if s in ids and s not in library:library[s]=Cortex(b,s,rec,pose['bodyPoses'][b])
        owners=[library[s] for s in ids]
        y=np.array(row['world_route_m'])*1000;start=y.copy();edges=[];seeds=[];intervals=[];normals=[]
        for p in y[1:-1]:
            options=[(*o.feature(p),o.sid,o) for o in owners]
            distance,i,t,sid,o=min(options,key=lambda item:item[0])
            if distance>3e-5:raise ValueError(f'No registered cortical edge at a bend: {distance} mm')
            pair=o.edges[i];incident=np.flatnonzero(o.inverse==i)//3
            edges.append(dict(stable_id=sid,body=o.body,source_edge=o.first[pair].tolist(),incident_faces=incident.tolist()))
            intervals.append(o.verts[i]);normals.append(o.normal[incident]);seeds.append(t)
        E=np.array(intervals);delta=E[:,1]-E[:,0]
        def evaluate(t):
            y[1:-1]=E[:,0]+t[:,None]*delta
            L,d=line_length(y)
            return L,np.einsum('ij,ij->i',d,delta)
        sol=minimize(evaluate,seeds,method='L-BFGS-B',jac=True,bounds=[(0,1)]*len(seeds),options={'gtol':1e-9,'ftol':2e-15,'maxiter':1800,'maxls':80,'maxcor':20})
        L,g=evaluate(sol.x);_,d=line_length(y)
        for k,(info,t,n) in enumerate(zip(edges,sol.x,normals)):
            if min(t,1-t)<1e-6:
                o=library[info['stable_id']];ve=info['source_edge'][int(t>.5)];node=np.flatnonzero(o.first==ve)[0]
                n=o.normal[(o.faces==node).any(1)]
            weight,err=nnls(n.T,d[k]);info.update(weight=float(t),normal_cone_residual_per_unit_tension=float(err),reaction_per_unit_tension=d[k].tolist())
        sr=verify(start,owners);er=verify(y,owners)
        rec=dict(fiber=row['name'],section=digest(fi),material_donor=digest(donor),registered_bones=digest(bone),open_banks=[o.sid for o in owners if o.solid is None],original_interiors=sr,relaxed_interiors=er,edge_tolerance_mm=3e-5,segment_tolerance_mm=1e-5,slack_m=row['reference_m'],original_length_m=line_length(start)[0]*.001,relaxed_length_m=.001*L,world_m=(y*.001).tolist(),edges=edges,solver=sol.message,max_normal_cone_residual=max(e['normal_cone_residual_per_unit_tension'] for e in edges))
        (output/fi.name).write_text(json.dumps(rec,indent=2)+'\n');print(row['name'],round(L,3),len(sr),len(er),round(rec['max_normal_cone_residual'],5),flush=True)
if __name__=='__main__':run(*[Path(k).resolve() for k in sys.argv[1:]])
