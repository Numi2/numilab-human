"""Refit the native constitutive limiter witness and its complete incident stencil."""
from itertools import combinations, product
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

root=Path(__file__).resolve().parent
source=root/"absolute-volume-repaired-tetgen"
points_path=source/"mesh.node"
cells_path=source/"mesh.ele"
x=np.loadtxt(points_path,skiprows=1,usecols=(1,2,3),dtype=np.float32).astype(float)
cells=np.loadtxt(cells_path,skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32)
tets=cells[:,:4]-1
motion=json.loads((root/"floor-native-live-parent-018/compact-motion-evidence.json").read_text())
witness,=motion["constitutive_witness"]
rejected=witness["closest_rejected_trial"]
assert witness["object"]==15 and rejected["reason"]==8
assert rejected["tile_witness_tets"]==[rejected["first_bad_tet"]]
# Two registered ACLs and twelve fascia objects occupy 2163 elements.
# The single preceding aortic wall uses six tetrahedra per axial-sector prism.
skin_element_offset=2163+8*24*6
bad=np.array([rejected["first_bad_tet"]-skin_element_offset])
assert cells[bad[0],4]==16 and tets[bad[0]].tolist()==[106228,106237,106236,106229]
edges=tuple(combinations(range(4),2))
solution=x.copy()
def metric(y):
    e=y[:,1:]-y[:,:1]
    d=np.einsum("ij,ij->i",e[:,0],np.cross(e[:,1],e[:,2]))
    span=np.max([np.linalg.norm(y[:,a]-y[:,b],axis=1) for a,b in edges],axis=0)
    return d/6., d/span**3

result=[]
for nodes, required, max_m in ((tets[bad[0]],bad,5e-6),):
    patch=np.flatnonzero(np.any(np.isin(tets,nodes),axis=1))
    patch_nodes=tets[patch]
    points=x[patch_nodes]
    v0,q0=metric(points)
    target_v=np.maximum(np.minimum(.995*v0,5e-18),1.03e-18)
    target_v[~np.isin(patch,required)]=np.maximum(
        np.minimum(.97*v0[~np.isin(patch,required)],4e-18),1.01e-18)
    target_q=np.maximum(1.002e-5,np.minimum(.5*q0,5e-5))
    selected=np.isin(patch,required)
    target_v[selected]=12e-18
    target_q[selected]=4e-4
    scale=np.maximum(v0,1.e-18)
    i,j=np.nonzero(np.isin(patch_nodes,nodes))
    k=np.array([np.flatnonzero(nodes==n)[0] for n in patch_nodes[i,j]])
    def trial(u):
        p=points.copy()
        p[i,j]+=u.reshape(-1,3)[k]*1e-6
        return p
    def constraints(u):
        v,q=metric(trial(u))
        margin = np.where(selected, 1.20, 1.0)
        return np.r_[(v-margin*target_v)/scale, (q-margin*target_q)/np.maximum(q0,1e-5)]
    def jacobian(u):
        y=trial(u)
        e=y[:,1:]-y[:,:1]
        c=np.cross(e[:,[1,2,0]],e[:,[2,0,1]])
        grad=np.concatenate((-c.sum(axis=1)[:,None],c),axis=1)
        d=np.einsum("ij,ij->i", e[:,0], c[:,0])
        ab=np.asarray(edges)
        offsets=y[:,ab[:,0]]-y[:,ab[:,1]]
        squares=np.sum(offsets*offsets,axis=2)
        longest=squares.argmax(axis=1)
        rr=np.arange(len(y))
        length2=squares[rr,longest]
        den=length2**1.5
        grad_den=np.zeros_like(grad)
        slope=3.*den[:,None]*offsets[rr,longest]/length2[:,None]
        grad_den[rr,ab[longest,0]]=slope
        grad_den[rr,ab[longest,1]]=-slope
        dv=np.zeros((len(y), len(nodes), 3))
        dq=np.zeros_like(dv)
        dv[i,k] = grad[i,j]*(1e-6/(6.*scale[i,None]))
        dq[i,k] = (grad[i,j] - (d/den)[i,None]*grad_den[i,j]) * (
            1e-6/(den[i,None]*np.maximum(q0[i,None],1e-5)))
        return np.vstack((dv.reshape(len(y),-1),dq.reshape(len(y),-1)))
    def cost(u):
        return .5*(u@u)
    opt=minimize(cost, np.zeros(3*len(nodes)), jac=lambda u:u, method="SLSQP",
        constraints=[{"type":"ineq","fun":constraints,"jac":jacobian}],
        bounds=[(-max_m*1e6,max_m*1e6)]*(3*len(nodes)),
        options={"ftol":1e-11,"maxiter":150})
    new=(x[nodes]+opt.x.reshape(-1,3)*1e-6).astype(np.float32).astype(float)
    rounded=points.copy()
    rounded[i,j]=new[k]
    v,q=metric(rounded)
    if not (np.all(v>=target_v) and np.all(q>=target_q)):
        # Solver displacements can be below the next representable world
        # coordinate. Select the nearest admissible actual Float32 emission.
        original32=x[nodes].astype(np.float32)
        modified=None
        score=np.inf
        for count in range(1,4):
            for indices in combinations(range(original32.size),count):
                for signs in product((-1,1),repeat=count):
                    candidate=original32.copy()
                    idx=np.array(indices)
                    destination=np.array(signs,dtype=np.float32)*np.inf
                    candidate.ravel()[idx]=np.nextafter(candidate.ravel()[idx],destination)
                    new_score=np.sum((candidate.astype(float)-x[nodes])**2)
                    if new_score>=score:
                        continue
                    trial_points=points.copy()
                    trial_points[i,j]=candidate[k]
                    vt,qt=metric(trial_points)
                    if np.all(vt>=target_v) and np.all(qt>=target_q):
                        score=new_score
                        modified=(candidate.astype(float),vt,qt)
            if modified is not None:
                new,v,q=modified
                break
    print("patch", required.tolist(),opt.message,"min",v.min(),q.min(),"offset um",np.linalg.norm(opt.x),"constraint",constraints(opt.x).min(),flush=True)
    assert np.all(v>1e-18) and np.all(q>=1e-5)
    assert np.all(v[selected]>=target_v[selected]) and np.all(q[selected]>=target_q[selected])
    displacement=np.linalg.norm(new-x[nodes],axis=1)
    assert displacement.max()<=max_m
    solution[nodes]=new
    result.append({"target_cell_rows":required.tolist(),"affected_cells":patch.tolist(),
        "moved_node_rows":nodes.tolist(),"max_move_m":float(displacement.max()),
        "selected_old_volume_m3":v0[selected].tolist(),"selected_new_volume_m3":v[selected].tolist(),
        "selected_old_condition":np.linalg.cond(points[selected,1:]-points[selected,:1]).tolist(),
        "selected_new_condition":np.linalg.cond(rounded[selected,1:]-rounded[selected,:1]).tolist(),
        "min_volume_m3":float(v.min()),"min_normalized_volume":float(q.min()),
        "maximum_cell_volume_change_m3":float(np.max(abs(v-v0))),
        "material_volume_change_m3":{str(m):float((v-v0)[cells[patch,4]==m].sum())
                                      for m in np.unique(cells[patch,4])}})
out=root/"constitutive-witness-repaired-tetgen"
out.mkdir(exist_ok=False)
# Preserve the actual element bytes and row identities.
(out/"mesh.ele").hardlink_to(cells_path)
with points_path.open() as prior,(out/"mesh.node").open("w") as stream:
    stream.write(prior.readline())
    changes=set(np.flatnonzero(np.any(x!=solution,axis=1)))
    for row,line in enumerate(prior):
        if row in changes:
            stream.write(str(row+1)+" "+" ".join(format(a,".17g") for a in solution[row])+"\n")
        else:
            stream.write(line)
def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
receipt={"status":"PASS_LOCAL_ABSOLUTE_VOLUME_SCREEN","nodes":len(solution),"cells":len(tets),
         "mesh_node_sha256":sha(out/"mesh.node"),"mesh_ele_sha256":sha(out/"mesh.ele"),
         "source_node_sha256":sha(points_path),"patches":result,"crossing_checks":0,
         "native_steps":0,"physical_admission":False}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
