"""Compute stretch already prescribed by protected vertex pairs, triangles, or entire tetrahedra."""
import json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def run(job,trial,out):
    root=job/'interval-certified-scene/material';x=np.loadtxt(root/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    y=np.loadtxt(trial/'current.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    rows=np.loadtxt(root/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=rows[:,:4]-1
    z=np.load(job/'registered-boundary/registered-boundary.npz');ct=np.load(job/'deep-fat-cortex/registered-cortical-facets.npz')
    pinned=np.zeros(len(x),bool);pinned[z['O_node_joint']]=1;pinned[ct['cell_side_material_bank_nodes'][:,-3:].ravel()]=1
    a=json.loads((job.parent/'ligament-equilibrium-070/scene-registration/registration.json').read_bytes())
    pinned[a['first_contact_new']+[w[2] for w in a['anchors']]]=1
    previously=pinned.copy();pinned[np.load(job/'held-loaded-cortex/nodes.npy')]=1
    fails=[]
    for beg in range(0,len(t),32768):
        v=t[beg:beg+32768];st=np.linalg.svd(np.linalg.solve(x[v[:,1:]]-x[v[:,:1]],y[v[:,1:]]-y[v[:,:1]]),compute_uv=False);fails.extend((beg+np.flatnonzero((st[:,0]>2)|(st[:,2]<.5))).tolist())
    d={}
    for name,mask in [('outer_skin_cortical_and_attachment_coordinates',previously),('including_neighborhood_of_measured_bone_overlaps',pinned)]:
        counts={15:0,16:0};worst={}
        for i in fails:
            v=t[i][mask[t[i]]]
            if len(v)<2:continue
            # On the span of the fixed reference edges the affine map is
            # prescribed exactly. Its extremal stretches bound those of any
            # three-dimensional extension regardless of free vertex choices.
            A=(x[v[1:]]-x[v[0]]).T;B=(y[v[1:]]-y[v[0]]).T
            Q,R=np.linalg.qr(A,mode='reduced');M=np.linalg.solve(R.T,B.T).T
            sv=np.linalg.svd(M,compute_uv=False)
            if sv[0]>2 or sv[-1]<.5:
                m=int(rows[i,4]);counts[m]+=1
                val=max(float(sv[0]),1/float(sv[-1]))
                if m not in worst or val>worst[m]['severity']:worst[m]=dict(cell=i,protected_nodes=v.tolist(),severity=val,minimum_prescribed_stretch=float(sv[-1]),maximum_prescribed_stretch=float(sv[0]))
        d[name]=dict(cells_with_prescribed_severe_stretch=counts,worst=worst)
    out.write_text(json.dumps(dict(pose=digest(trial/'current.node'),mesh=digest(root/'mesh.ele'),severe_cells=len(fails),bounds=d),indent=2)+'\n')
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
