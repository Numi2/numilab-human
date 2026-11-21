"""Independently re-solve material contact at each supplied native pose on SSH Numi."""
from pathlib import Path
import copy,json,sys
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.spatial.transform import Rotation
from shoulder_inputs import digest, old
from source_pair_audit import inputs
from audit_solids import winding
from numilab_human.capsular_material_registration import read_material_banks, emitted_body_vertices

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)

class Bone:
    def __init__(self, mesh, pose, sid):
        points, faces = mesh[:2]
        verts, self.representative, inv = np.unique(points, axis=0, return_inverse=True, return_index=True)
        self.sid = sid
        self.v = pose[:3] + Rotation.from_quat(pose[3:]).apply(verts)
        self.f = inv[faces]
        tri = self.v[self.f]
        self.tri = tri
        self.normals = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        edges = np.stack((self.f[:, [0, 1]], self.f[:, [1, 2]], self.f[:, [2, 0]]),axis=1)
        self.edges, mapping, n = np.unique(np.sort(edges.reshape(-1, 2), axis=1),axis=0,return_inverse=True,return_counts=True)
        if not np.all(n==2):
            raise ValueError(f'Bone {sid} is not a closed two-manifold after identifying exact seams')
        self.fe = mapping.reshape(-1,3)
        self.ef = (np.argsort(mapping,kind='stable')//3).reshape(-1,2)
        centered=tri-self.v.mean(0)
        volume=np.einsum('ij,ij->i',centered[:,0],np.cross(centered[:,1],centered[:,2])).sum()/6
        self.normals *= np.sign(volume)

def tauten(bones,ends,edges):
    lookup={b.sid:b for b in bones}
    supports=[lookup[s].v[lookup[s].edges[e]] for s,e,t in edges]
    if not supports:
        return np.asarray(ends),np.array([])
    supports=np.asarray(supports)
    a=supports[:,0];delta=supports[:,1]-a
    def points(t):
        return np.vstack((ends[0],a+t[:,None]*delta,ends[1]))
    def f(t):
        d=np.diff(points(t),axis=0)
        u=unit(d)
        return np.linalg.norm(d,axis=1).sum()*1000, np.sum((u[:-1]-u[1:])*delta,axis=1)*1000
    result=minimize(f,[e[2] for e in edges],method='L-BFGS-B',jac=True,bounds=[(0,1)]*len(edges),
        options=dict(maxiter=180,maxcor=12,ftol=2e-15,gtol=2e-9))
    # L-BFGS can stop on a sub-ulp change in length with a measurable
    # transverse force on a short edge. Polish the same constrained length
    # using its exact edge-coordinate Hessian and active bounds.
    x=result.x.copy()
    for _ in range(14):
        d=np.diff(points(x),axis=0)
        lengths=np.linalg.norm(d,axis=1)
        if lengths.min()<1e-9:break
        u=d/lengths[:,None]
        grad=np.sum((u[:-1]-u[1:])*delta,axis=1)
        free=~(((x<=1e-10)&(grad>=0))|((x>=1-1e-10)&(grad<=0)))
        if not free.any() or np.max(abs(grad[free])/np.linalg.norm(delta[free],axis=1))<1e-10:break
        K=(np.eye(3)[None,:,:]-u[:,:,None]*u[:,None,:])/lengths[:,None,None]
        diag=np.einsum('ni,nij,nj->n',delta,K[:-1]+K[1:],delta)
        off=-np.einsum('ni,nij,nj->n',delta[:-1],K[1:-1],delta[1:])
        H=np.diag(diag)+np.diag(off,1)+np.diag(off,-1)
        try: dx=np.linalg.solve(H[free][:,free],-grad[free])
        except np.linalg.LinAlgError:break
        if not np.all(np.isfinite(dx)) or np.dot(grad[free],dx)>=0:break
        improved=False
        for fraction in (1.,.5,.25,.125,.0625):
            trial=x.copy();trial[free]=np.clip(x[free]+fraction*dx,0,1)
            if np.max(abs(trial-x))<1e-14:break
            if np.linalg.norm(np.diff(points(trial),axis=0),axis=1).sum()<=lengths.sum()+1e-17:
                x=trial;improved=True;break
        if not improved:break
    return points(x),x

def certify(bones,points,edges,t,ends):
    lookup={b.sid:b for b in bones};d=np.diff(points,axis=0);u=unit(d)
    gr=np.vstack((-u[0],u[:-1]-u[1:],u[-1]))
    hits=[];inside=[];constraints=[];reactions=[]
    if np.min(np.linalg.norm(d,axis=1))<1.e-9:
        raise ValueError("Coincident material reaction knots")
    for bone in bones:
        tri=bone.tri
        lo,hi=tri.min(1),tri.max(1)
        # Recognize a shared source face by its material features. Only an
        # exact edge/vertex or the actual barycentric attachment can claim it.
        features=[]
        for idx in range(len(points)):
            if idx==0 or idx==len(points)-1:
                endpoint = ends[0 if idx==0 else 1]
                features.append({endpoint['triangle']} if endpoint['stable']==bone.sid else set())
                continue
            sid,e,_=edges[idx-1]; pos=t[idx-1]
            if sid!=bone.sid:
                features.append(set());continue
            if pos<=1e-12 or pos>=1-1e-12:
                v=bone.edges[e,int(pos>=1-1e-12)]
                features.append(set(np.flatnonzero(np.any(bone.f==v,axis=1))))
            else:
                features.append(set(bone.ef[e]))
        for j,(a,b) in enumerate(zip(points[:-1],points[1:])):
            common=features[j]&features[j+1]
            mask=np.all((lo<=np.maximum(a,b)+1e-9)&(hi>=np.minimum(a,b)-1e-9),axis=1)
            ids=np.flatnonzero(mask)
            for fraction,k in old.audit.lp.vector_hits(a,b,(bone.v,bone.f[ids])):
                face=ids[k]
                if face not in common:
                    hits.append((j,bone.sid,int(face),fraction))
            if common:
                # A triangle is convex. Reconstructing both material knots
                # in that face certifies the whole span, including its middle.
                for f in common:
                    if max(abs((p-tri[f,0])@bone.normals[f]) for p in (a,b))>2e-13:
                        raise ValueError("Shared-face reconstruction differs from the stored surface")
            elif abs(winding(tri,(a+b)*.5))>.5:
                inside.append((j,bone.sid))
    for i,((sid,edge,_),v) in enumerate(zip(edges,t)):
        bone=lookup[sid];a,b=bone.edges[edge]
        faces=bone.ef[edge]
        if v<1e-6 or v>1-1e-6:
            vertex=a if v<1e-6 else b
            faces=np.flatnonzero(np.any(bone.f==vertex,axis=1))
        vector=gr[i+1]
        _,res=nnls(bone.normals[faces].T,vector)
        constraints.append(float(res));reactions.append(float(np.linalg.norm(vector)))
    return dict(length_m=float(np.linalg.norm(d,axis=1).sum()),edges=len(edges),crossings=hits,inside=inside,
        normal_cone_residual_per_unit_tension=max(constraints,default=0.),contact_reaction_per_unit_tension=reactions,
        shortest_segment_m=np.linalg.norm(d,axis=1).min(),
        force_residual_per_unit_tension=float(np.linalg.norm(gr.sum(0))),
        moment_residual_m_per_unit_tension=float(np.linalg.norm(np.cross(points-points[0],gr).sum(0))))

def coords(bones,ends,edges):
    points=[ends[0]]
    for sid,e,t in edges:
        b=bones[sid];a,c=b.v[b.edges[e]]
        points.append((1-t)*a+t*c)
    return np.vstack([*points,ends[1]])

def size(pts):
    return np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()

def vertex(bones,k):
    sid,e,t=k
    ij=bones[sid].edges[e]
    if t<1e-7:return sid,ij[0]
    if t>1-1e-7:return sid,ij[1]

def simplify(bones,ends,edges):
    result=[]
    for k in edges:
        v=vertex(bones,k)
        if result and v is not None and v==vertex(bones,result[-1]):continue
        result.append(k)
    return result

def optimize(bones,ends,edges):
    for i in range(8):
        x,t=tauten(list(bones.values()),ends,edges)
        updated=[(s,e,float(t0)) for (s,e,_),t0 in zip(edges,t)]
        short=simplify(bones,ends,updated)
        if len(short)==len(updated):return updated
        edges=short
    return edges

def other_solids(points,pose,cat,meshes,owners):
    lo=points.min(0);hi=points.max(0);reports=[]
    for sid,(v,f,n,lower,upper) in meshes.items():
        if sid in owners:continue
        q=pose[cat[sid]['body']]
        local=Rotation.from_quat(q[3:]).inv().apply(points-q[:3])
        if np.any(lower>local.max(0)+1e-9) or np.any(upper<local.min(0)-1e-9):continue
        tri=v[f];front=tri.min(1);back=tri.max(1)
        for j,(a,b) in enumerate(zip(local,local[1:])):
            if np.any(lower>np.maximum(a,b)+1e-9) or np.any(upper<np.minimum(a,b)-1e-9):continue
            mask=np.all((front<=np.maximum(a,b)+1e-9)&(back>=np.minimum(a,b)-1e-9),axis=1)
            for t,k in old.audit.lp.vector_hits(a,b,(v,f[mask])):
                reports.append([j,sid,'cross',int(np.flatnonzero(mask)[k]),t])
            if abs(winding(tri,(a+b)/2))>.5:reports.append([j,sid,'inside'])
    return reports

def virtual_work(q,p,edges,row,direction,templates):
    ends=row['attachments'];sids=[e['stable'] for e in ends];sid=sids[1];body=ends[1]['body']
    pts=p;u=unit(np.diff(pts,axis=0));g=np.vstack((-u[0],u[:-1]-u[1:],u[-1]))
    owned=np.array([sids[0],*(e[0] for e in edges),sids[1]])==sid
    dq,omega=direction
    velocity=dq+np.cross(omega,pts[owned]-q[body][:3])
    expected=float(np.sum(g[owned]*velocity))
    # An independent length solve on both sides of the perturbation verifies
    # the envelope derivative. Accumulate forces at every actual contact.
    lengths=[]
    for sign in [-1,1]:
        h=sign*2e-6
        new={i:q[i].copy() for i in [e['body'] for e in ends]}
        new[body][:3]+=h*dq
        new[body][3:]=(Rotation.from_rotvec(h*omega)*Rotation.from_quat(new[body][3:])).as_quat()
        bp={};ep=[]
        for e in ends:
            v=new[e['body']];bp[e['stable']]=posed(templates[e['stable']],v)
            ep.append(v[:3]+Rotation.from_quat(v[3:]).apply(e['local_m']))
        sol=optimize(bp,ep,edges)
        lengths.append(np.linalg.norm(np.diff(coords(bp,ep,sol),axis=0),axis=1).sum())
    return dict(analytic=expected,central_difference=(lengths[1]-lengths[0])/4e-6)

def posed(template,q):
    b=copy.copy(template);rot=Rotation.from_quat(q[3:])
    b.v=q[:3]+rot.apply(template.v)
    b.tri=b.v[b.f];b.normals=rot.apply(template.normals)
    return b


def run(launch,anchors,motion,bone,sites,receipt,branches,out):
    _,poses,rest,_=inputs(launch,anchors,motion)
    poses[0]=rest
    poses[-4]={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in json.loads(receipt.read_text())['accepted_body_poses']}
    old.audit.ag.BONES=bone
    _,raw,_,_,_,_,_,cat,_,vo,io=old.audit.ag.read_payloads()
    meshes=old.load_local_meshes(raw,cat,vo,io)
    _,_,records,source_vertices,_=read_material_banks(bone)
    for sid,record in records.items():
        first,n=record[1:3];v=emitted_body_vertices(source_vertices[first:first+n],record)
        _,f,norm,_,_=meshes[sid];meshes[sid]=(v,f,norm,v.min(0),v.max(0))
    site_doc=json.loads(sites.read_text())
    if site_doc['candidate_sha256']!=digest(bone):raise ValueError("Attachment payload changed")
    named={r['side']:r for r in site_doc['rows']}
    templates={e['stable']:Bone(meshes[e['stable']],np.array([0,0,0,0,0,0,1.]),e['stable'])
               for r in named.values() for e in r['attachments']}
    rows=[]
    for branch in json.loads(branches.read_text()):
        q=poses[branch['step']];row=named[branch['side']]
        bp={};ends=[]
        for e in row['attachments']:
            sid=e['stable'];p=q[e['body']];bp[sid]=posed(templates[sid],p)
            actual=np.array(e['barycentric'])@meshes[sid][0][meshes[sid][1][e['triangle']]]
            if np.linalg.norm(actual-e['local_m'])>1e-12:raise ValueError("Material attachment disagrees with binary")
            ends.append(p[:3]+Rotation.from_quat(p[3:]).apply(actual))
        ed=optimize(bp,ends,branch['edges'])
        pts=coords(bp,ends,ed)
        c=certify(list(bp.values()),pts,ed,[k[2] for k in ed],row['attachments'])
        hits=other_solids(pts,q,cat,meshes,bp)
        good=(not(c['crossings'] or c['inside'] or hits) and c['normal_cone_residual_per_unit_tension']<1e-6)
        checks=[]
        if good:
            for a in np.eye(6):
                checks.append(virtual_work(q,pts,ed,row,(a[:3],a[3:]),templates))
        error=[abs(w['analytic']-w['central_difference']) for w in checks]
        good=good and max(error[:3],default=0.)<1e-5 and max(error[3:],default=0.)<1e-6
        rows.append(dict(step=branch['step'],side=branch['side'],branch=branch['branch'],admitted=good,
                    length_m=c['length_m'],material_edge_count=len(ed),crossings=c['crossings'],inside=c['inside'],
                    other_bones=hits,normal_cone_error=c['normal_cone_residual_per_unit_tension'],
                    net_force_error_per_N=c['force_residual_per_unit_tension'],
                    net_moment_error_Nm_per_N=c['moment_residual_m_per_unit_tension'],
                    virtual_work_translation_error=error[:3],virtual_work_rotation_error_m=error[3:]))
    out.write_text(json.dumps(dict(bones_sha256=digest(bone),sites_sha256=digest(sites),rows=rows),
        indent=2,allow_nan=False,default=lambda v:v.item())+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
