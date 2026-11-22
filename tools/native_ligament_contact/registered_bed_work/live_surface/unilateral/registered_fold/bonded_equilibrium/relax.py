"""Prepare a loaded pose by relaxing measured bonded tissue stars at fixed exterior geometry."""
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.linalg import qr
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from admissible import material_direction, emit_conservative
from cavity_mass import cavity_delta
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from refine_posterior_contact import mechanical
from supported_material_admission import certify, installed_topology, material_state, stress_limits

def read(path):
    return json.loads(path.read_text())

def stiffness(reference, current, cells, free, roles, props):
    """Positive material metric. The search and convergence use the actual elastic force."""
    n=len(free)
    local=np.full(len(current),-1,dtype=int)
    local[free]=np.arange(n)
    ids=local[cells]
    R=reference[cells[:,1:]]-reference[cells[:,:1]]
    Y=current[cells[:,1:]]-current[cells[:,:1]]
    invR=np.linalg.inv(R).transpose(0,2,1)
    invY=np.linalg.inv(Y).transpose(0,2,1)
    G=np.concatenate((-invR.sum(1,keepdims=True),invR),axis=1)
    H=np.concatenate((-invY.sum(1,keepdims=True),invY),axis=1)
    volumes=np.linalg.det(R)/6
    mu,bulk=(volumes[:,None]*props[roles-15]).T
    blocks=[]; rows=[]; cols=[]
    axes=np.arange(3)
    for i in range(4):
        for j in range(4):
            use=(ids[:,i]>=0)&(ids[:,j]>=0)
            if not use.any():continue
            g=(mu[use]*np.sum(G[use,i]*G[use,j],axis=1))[:,None,None]*np.eye(3)
            g+=bulk[use,None,None]*H[use,i,:,None]*H[use,j,None,:]
            ri=3*ids[use,i,None]+axes; cj=3*ids[use,j,None]+axes
            blocks.append(g.ravel())
            rows.append(np.broadcast_to(ri[:,:,None],g.shape).ravel())
            cols.append(np.broadcast_to(cj[:,None,:],g.shape).ravel())
    return coo_matrix((np.concatenate(blocks),(np.concatenate(rows),np.concatenate(cols))),shape=(3*n,3*n)).tocsc()

def relax(root, constrained=False, release_witness=False):
    out=root/'bonded-equilibrium'
    if release_witness:
        out=out/'free-interface';out.mkdir(exist_ok=True)
    run=root/'fold-repair/accepted-2'; ledger=root/'fold-repair/material'; owner=root/'material-direction/release/owner'
    e=read(run/'launch.json')['environment']
    X,Y,src,source_roles,pins=inputs(run)
    X,Y=X.astype(float),Y.astype(float)
    t,roles=installed_topology(src,source_roles,e)
    labels=ledger/'mesh/current-child.labels.csv'
    if sha(labels)!=read(labels.with_suffix('.json'))['labels_sha256']:raise ValueError('Parent labels changed')
    raw_labels=np.loadtxt(labels,delimiter=',',skiprows=1,usecols=(0,1,2),dtype='i4')
    if not np.array_equal(raw_labels[:,0],np.arange(len(src))) or not np.array_equal(raw_labels[:,1],source_roles):raise ValueError('Material label order')
    flow=ledger/'fine-mass-routed/current-density-closed-flow.npz'
    if sha(flow)!=read(flow.parent/'mass-transfer.json')['flow_sha256']:raise ValueError('Material mass identity')
    with np.load(flow) as f:
        old_mass=f['element_mass_kg']
    with np.load(e['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']) as f:
        retained=f['retained_input_cell_rows']
        parents=f['replacement_parent_cell_rows'][f['replacement_parent_offsets'][:-1]]
        cell_mass=np.r_[old_mass[retained],f['replacement_mass_kg']]
        donor=np.r_[raw_labels[retained,2],raw_labels[parents,2]]
        protected=np.unique(np.r_[f['replacement_tetrahedra'].ravel(),src[np.setdiff1d(np.arange(len(src)),retained)].ravel()])
        if release_witness:
            delta_mass,cavity_nodes,cavity_mass=cavity_delta(src,old_mass,retained,f['replacement_tetrahedra'],f['replacement_mass_kg'],len(Y))
    props,matfiles=materials(owner); limits=stress_limits(matfiles)
    endpoint=run/'physical-accepted.node'
    if sha(endpoint)!=read(run/'verified-endpoint.json')['physical_nodes_sha256']:raise ValueError('Endpoint identity')
    end=np.loadtxt(endpoint,skiprows=1,usecols=(1,2,3))
    layer=np.full(len(Y),-1,dtype='i1')
    mp=Path(e['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if sha(mp)!=e['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_SHA256']:raise ValueError('Layer identity')
    with mp.open() as f:
        for row in csv.DictReader(f):
            node=int(row['joint_node_id'])-1; val={'':0,'O':1,'I':2,'D':3}[row['registered_surface_layer']]
            if layer[node] not in (-1,val):raise ValueError('Conflicting material interface map')
            layer[node]=val
    if np.any(layer<0):raise ValueError('Unregistered node')
    tab=re.search(r'constexpr Anchor anchors\[\]=\{(.*?)\};',(owner/'NumiHumanLoadedSkinPoseData.inc').read_text(),re.S)
    anchors=np.asarray(re.findall(r'\{\s*\d+u?\s*,\s*(\d+)u?\s*,',tab[1]),dtype=int)
    if len(anchors)!=14:raise ValueError('Anchor ownership changed')
    # Take the complete star of every loaded bed vertex and every observed strain failure.
    # Outer and deep geometry, entheses and remeshed cavity boundaries are never variables.
    width=float.fromhex('0x1.4f8b580000000p-17')
    bed=np.flatnonzero((layer==1)&(np.minimum(Y[:,2],end[:,2])<width))
    severe=[]
    for i in range(0,len(t),32768):
        c=t[i:i+32768];_,s,_=material_state(X[c],end[c],props[roles[i:i+len(c)]-15])
        severe.extend((i+np.flatnonzero((s[:,0]>1.5)|(s[:,-1]<2/3))).tolist())
    offsets=read(root/'contact-pockets/object-offset.json')
    rejection=read(root/'material-direction/release/result.json')
    row=rejection['native_cell']-offsets['source_element_offset']
    if retained[row]!=rejection['source_cell']:raise ValueError('Wrong native tissue row')
    seeds=np.unique(np.r_[bed,t[severe].ravel(),t[row]])
    first=np.flatnonzero(np.isin(t,seeds).any(1))
    # Moving an existing loaded vertex advects its mass. Remeshing protection
    # governs the reference topology. Permit the four measured interface nodes
    # only when their full installed stars and replacement masses are evaluated.
    allowed=t[row] if release_witness else np.empty(0,dtype=int)
    if release_witness and not np.all(layer[allowed]==2):raise ValueError('Witness is not an entirely bonded cell')
    free=np.setdiff1d(np.unique(t[first]),np.r_[anchors,np.setdiff1d(protected,allowed)])
    free=free[np.isin(layer[free],[0,2])]
    incident=np.flatnonzero(np.isin(t,free).any(1))
    nodes=np.unique(t[incident]); cells=np.searchsorted(nodes,t[incident])
    f=np.searchsorted(nodes,free); x=X[nodes]; y=Y[nodes]; tags=roles[incident]
    # Restrict each original anatomical donor/material's first moment on the
    # actual mass map. Reference density and all element masses stay unchanged.
    gids,group=np.unique(np.c_[tags,donor[incident]],axis=0,return_inverse=True)
    weights=np.zeros((len(gids),len(nodes)))
    for k in range(4):np.add.at(weights,(group,cells[:,k]),cell_mass[incident]/4)
    m=weights.sum(0)
    W=weights[:,f]
    if release_witness:
        # The native cavity check compares both lumpings in the new loaded
        # pose. Conserve that independently from the actual donor moments.
        radius=np.max(np.abs(Y[cavity_nodes]-Y[cavity_nodes[0]]))
        budget=1e-10*cavity_mass*radius
        old_change=delta_mass[cavity_nodes]@(Y[cavity_nodes]-Y[cavity_nodes[0]])
        reserve=.5*(budget-np.max(abs(old_change)))
        if reserve<=0:raise ValueError('No native loaded moment headroom')
        W=np.vstack((W,delta_mass[free]*(8e-11/reserve)))
    normalized=W/(np.linalg.norm(W,axis=1,keepdims=True)+1e-100)
    Q,R,p=qr(normalized.T,mode='economic',pivoting=True)
    diagonals=abs(np.diag(R));keep=diagonals>diagonals.max()*1e-11
    W0=Q[:,keep].T
    C=np.kron(W0,np.eye(3))
    def evaluate(pos):
        en,force=mechanical((x,pos),cells,tags,props)
        force[:,2]-=9.81*m
        return en+9.81*(m@(pos[:,2]-y[:,2])),force[f]
    def assess(pos):
        en,force=evaluate(pos); projected=force-W0.T@(W0@force)
        return dict(potential_J=en,free_force_l2_N=float(np.linalg.norm(force)),
                    constrained_free_force_l2_N=float(np.linalg.norm(projected)),
                    max_unbalanced_force_N=float(np.linalg.norm(force,axis=1).max()),
                    source_moment_drift_kg_m=float(abs(W@(pos[f]-y[f])).max()),
                    admission=certify(x,pos,cells,tags,props,limits))
    start=assess(y); pos=y.copy()
    if not start['admission']['passed']:raise ValueError('Starting material region inadmissible')
    history=[]
    for it in range(28 if constrained else 16):
        en,force=evaluate(pos)
        if constrained:
            delta=material_direction(x,pos,cells,f,force,C)
        else:
            A=stiffness(x,pos,cells,f,tags,props)
            solver=splu(A)
            uncon=solver.solve(force.ravel())
            inverse=solver.solve(C.T)
            delta=(uncon-inverse@np.linalg.solve(C@inverse,C@uncon)).reshape(-1,3)
        slope=np.sum(delta*force); best=None
        if slope<=0:break
        for back in range(26):
            step=2.**-back
            test=pos.copy();test[f]+=step*delta
            emitted=emit_conservative(test,y,f,W) if constrained else test.astype('f4').astype(float)
            edges=emitted[cells[:,1:]]-emitted[cells[:,:1]]
            if np.any(np.linalg.det(edges)<=0) or not certify(x,emitted,cells,tags,props,limits)['passed']:continue
            a=assess(emitted)
            if a['source_moment_drift_kg_m']>8e-11 or a['potential_J']>en-.0001*step*slope:continue
            best=(emitted if constrained else test,a,step)
            break
        if best is None:break
        pos,a,step=best;history.append(dict(iteration=it,fraction=step,potential_J=a['potential_J'],projected_force_N=a['constrained_free_force_l2_N']))
        print(history[-1],flush=True)
        if np.linalg.norm(step*delta)<1e-10:break
    final=pos.astype('f4').astype(float);after=assess(final)
    used=(final[f]!=y[f]).any(1);moved=free[used]
    if len(moved):
        np.savez_compressed(out/'preload.npz',rows=moved,coordinates=final[f[used]].astype('f4'))
    result=dict(source=pins,endpoint=sha(endpoint),materials=matfiles,registered_mass=sha(flow),registered_map=sha(mp),witness_source_row=int(retained[row]),bed_nodes=len(bed),strain_witnesses=len(severe),free_bonded=int(sum(layer[free]==2)),free_interior=int(sum(layer[free]==0)),closed_star_cells=len(cells),donor_material_rows=len(gids),anchors=len(anchors),largest_change_m=float(np.linalg.norm(final-y,axis=1).max()),moved=len(moved),initial=start,emitted=after,iterations=history)
    if release_witness:
        result['released_cavity_nodes']=allowed.tolist()
        result['moved_witness_nodes']=np.intersect1d(allowed,moved).tolist()
    (out/'fit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('fitted',len(free),len(cells),len(moved),start['free_force_l2_N'],after['free_force_l2_N'],flush=True)
if __name__=='__main__':relax(Path(sys.argv[1]).resolve(), '--constrained' in sys.argv[2:], '--free-interface' in sys.argv[2:])
