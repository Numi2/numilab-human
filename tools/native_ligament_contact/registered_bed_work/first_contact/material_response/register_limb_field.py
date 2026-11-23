"""Move each welded distal limb with its registered skeleton and solve a volumetric transition in its upper arm."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from rebind_ligament_material import banks
from audit_fiber_load_bounds import read_pose
from build_carpal_cavity import boundary

def tetra_jac(x,t):return x[t[:,1:]]-x[t[:,:1]]
def component(t,eligible,seed,n):
    ok=eligible[t].all(1);tet=t[ok];nodes,idx=np.unique(tet,return_inverse=True);ii=idx.reshape(-1,4)
    j=np.searchsorted(nodes,seed)
    if j==len(nodes) or nodes[j]!=seed:return np.zeros(n,bool)
    a=ii[:,[0,0,0,1,1,2]].ravel();b=ii[:,[1,2,3,2,3,3]].ravel()
    graph=coo_matrix((np.ones(len(a),'u1'),(a,b)),shape=(len(nodes),len(nodes))).tocsr()
    _,groups=connected_components(graph,directed=False);m=np.zeros(n,bool);m[nodes[groups==groups[j]]]=True
    return m
def shape(x,t):
    J=tetra_jac(x,t);v=np.linalg.det(J)/6
    length2=np.zeros(len(t))
    for i in range(3):
        for j in range(i+1,4):length2+=np.sum((x[t[:,i]]-x[t[:,j]])**2,axis=1)
    return v,12*np.cbrt(3*abs(v))**2/length2

def run(h,work,job,out):
    out.mkdir();r=h.parent;phys=h/'closed-native/native';source=json.load(open(job/'launch.json'))
    bones=next((r/'supported-bones').glob('*.nhbones'));base=Path(source['argv'][3]);before=banks(base);after=banks(bones)
    src=json.load(open(job/'native-initial-bodies.json'));poses={x['body_index']:x['current'] for x in src['all_bodies']}
    native=read_pose(r/'native-ligament-material/stderr.log')['bodyPoses'];xyz=np.loadtxt(phys/'mesh.node',skiprows=1,usecols=(1,2,3));z=np.loadtxt(phys/'mesh.ele',skiprows=1,dtype='i4');t=z[:,1:5]-1;mat=z[:,5]
    changed=np.zeros(len(xyz),bool);y=xyz.copy();stats={};anc=np.load(h/'scene-registration/node-map.npz')
    handwall=np.load(work/'wrist/cortical-facets.npz')['nodes'];pins=set()
    shoulder=np.load(h/'cortical-registration/registered-cortical-facets.npz')['cell_side_material_bank_nodes']
    fixed=set(shoulder[:,-3:].ravel());fixed.update(json.load(open(h/'scene-registration/registration.json'))['first_contact_new'])
    for side,b0,elbow,wrist in [('l',91,92,96),('r',41,42,46)]:
        old=poses[b0];new=native[b0];ori=Rotation.from_quat(old['orientation_world_xyzw']);target=Rotation.from_quat(new[3:])
        R=(target*ori.inv()).as_matrix();p=np.array(old['position_world_m']);dp=np.array(new[:3])-p
        move=lambda x:(x-p)@(R-np.eye(3)).T+dp
        largest=0
        for (b,s),(f,v,_) in before.items():
            if b<b0 or b>=b0+37:continue
            g,u,_=after[b,s];so=poses[b];sb=native[b]
            prior=Rotation.from_quat(so['orientation_world_xyzw']).apply(v)+so['position_world_m']
            now=Rotation.from_quat(sb[3:]).apply(u)+sb[:3]
            if not np.array_equal(f,g):raise ValueError('Changed limb bank topology')
            largest=max(largest,np.linalg.norm(now-prior-move(prior),axis=1).max())
        if largest>3e-6:raise ValueError(f'No shared limb transform {side} {largest}')
        # Humerus body origin is a COM. Recover geometric stations from its real
        # distal and proximal surface extents along the elbow-to-hand direction.
        a=np.array(poses[elbow]['position_world_m']);w=np.array(poses[wrist]['position_world_m'])
        axis=(a-p);axis/=np.linalg.norm(axis);station=(xyz-p)@axis
        sid=13 if b0==91 else 12
        v=ori.apply(before[b0,sid][1])+p;low,high=np.quantile((v-p)@axis,[.05,.95])
        seed=np.argmin(np.linalg.norm(xyz-w,axis=1))
        # Locate the continuous arm branch using physical tetrahedron connectivity.
        # The transition cannot consume torso, the opposite limb or new shoulder contacts.
        # Use a contralateral hand witness and the distal leg bound instead of a fraction
        # of total nodes: the hands carry most of the detailed mechanical surface.
        other= np.argmin(np.linalg.norm(xyz-np.array(poses[46 if side=='l' else 96]['position_world_m']),axis=1))
        for s in np.linspace(low,high+.095,17):
            arm=component(t,station>s,seed,len(xyz))
            if arm.sum()>500 and not arm[other] and xyz[arm,1].min()>w[1]-.25 and not (arm&changed).any():break
        else:raise ValueError('No isolated material branch')
        touched=arm[t].any(1);ta=t[touched];nodes,ix=np.unique(ta,return_inverse=True);tl=ix.reshape(-1,4)
        dist=np.maximum(0.,station[nodes]-s)
        width=max(.12,min(.24,float(high+.10-s)))
        anchor=(~arm[nodes])|(station[nodes]<=s+.003)|np.isin(nodes,list(fixed));rigid=arm[nodes]&(dist>=width)&~anchor
        if not rigid.any() or not anchor.any():raise ValueError('Empty transition or Dirichlet surface')
        interior=~(rigid|anchor);delta=np.zeros((len(nodes),3));delta[rigid]=move(xyz[nodes[rigid]])
        # The true FEM shape gradients strongly penalize shearing a thin cell.
        active=interior[tl].any(1);tr=tl[active];jac=tetra_jac(xyz[nodes],tr);V=np.linalg.det(jac)/6
        if np.any(V<=0):raise ValueError('Invalid source element')
        g=np.linalg.inv(jac).transpose(0,2,1);gr=np.concatenate((-g.sum(1)[:,None,:],g),1)
        ke=(gr@gr.transpose(0,2,1))*V[:,None,None]
        ii=np.broadcast_to(tr[:,:,None],ke.shape).ravel();jj=np.broadcast_to(tr[:,None,:],ke.shape).ravel()
        K=coo_matrix((ke.ravel(),(ii,jj)),shape=(len(nodes),len(nodes))).tocsc()
        unknown=np.flatnonzero(interior);rhs=-(K@delta)[unknown];S=K[unknown][:,unknown].tocsc();lu=splu(S)
        delta[unknown]=lu.solve(rhs);err=np.max(abs((K@delta)[unknown]))
        y[nodes]+=delta;changed[nodes[rigid|interior]]=True;v0,q0=shape(xyz[nodes],tl);v1,q1=shape(y[nodes].astype('f4').astype(float),tl)
        stats[side]=dict(rigid_inversions=int(np.sum((v1<=0)&rigid[tl].all(1))),transition_inversions=int(np.sum((v1<=0)&(~rigid[tl].all(1)))),box=[xyz[arm].min(0).tolist(),xyz[arm].max(0).tolist()],source_axis=axis.tolist(),seed=int(seed),connected=int(arm.sum()),rigid=int(rigid.sum()),transition=len(unknown),transition_elements=len(tr),station_m=float(s),transition_width_m=width,max_bank_error_m=largest,free_equation_residual=err,minimum_final_height_m=float(np.min(y[nodes,2])),min_source_height_m=float(np.min(xyz[nodes,2])),inverted=int((v1<=0).sum()),min_relative_volume=float(np.min(v1/v0)),source_min_shape=float(np.min(q0)),min_shape=float(np.min(q1)),rigid_wrist_nodes=bool(rigid[np.searchsorted(nodes,np.unique(handwall))].all()) if side=='l' and np.isin(handwall,nodes).all() else None)
        print(side,stats[side],flush=True)
    idx=np.flatnonzero(changed);np.savez_compressed(out/'material-transport.npz',nodes=idx,positions=y[idx].astype('f4'))
    report=dict(mesh=digest(phys/'mesh.node'),elements=digest(phys/'mesh.ele'),source_bones=digest(base),current_bones=digest(bones),source_pose=digest(job/'native-initial-bodies.json'),current_pose=digest(r/'native-ligament-material/stderr.log'),regions=stats,transport=digest(out/'material-transport.npz'))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
