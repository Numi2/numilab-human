"""Bound shared volume from below with disjoint interior balls of registered muscle cells."""
import json,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
from fast_skin_intersection_audit import owner,install_for_float32
from numilab_human.capsular_material_registration import read_material_banks,material_bank,emitted_body_vertices
from prepare_posterior_muscles import closed
from compare_hand_interfaces import query
from material_context import applied,sha
def measure(cfg,rc,cap,volume,exe,out):
    install_for_float32();args=json.loads(cfg.read_text())['argv'];bank=read_material_banks(Path(args[3]))
    evidence=json.loads(rc.read_text());fr={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in evidence['initial_anatomical_registration']['body_poses']}
    result={}
    for sid in (1,2,3,18,19):
        body=bank[2][sid][0];anchor,row,attr,f=material_bank(bank,sid,body)
        x=applied(fr[body],emitted_body_vertices(attr,row)).astype('f4')
        x,inv=np.unique(x,axis=0,return_inverse=True);f=inv[f];topology=closed(x,f)
        selfx=len(query(exe,x,f,np.ones(len(f),bool)))
        r=dict(owner=anchor['myosim_body'],topology=topology,nonadjacent_self_pairs=selfx,muscles={});result[sid]=r
        if any(topology.values()) or selfx:continue
        prepared=owner.prepare_float32_point_location_meters(x,f);tri=x[f].astype(float);low,high=tri.min(1),tri.max(1)
        for s in (29,30):
            capfile=cap/('entrance-%d.npz'%s);v=np.load(capfile)['xyz'].astype(float);t=np.loadtxt(volume/str(s)/'source.1.ele',skiprows=1,dtype=int)[:,1:5]
            p=v[t];A=np.stack([np.linalg.norm(np.cross(p[:,(k+2)%4]-p[:,(k+1)%4],p[:,(k+3)%4]-p[:,(k+1)%4]),axis=1) for k in range(4)],axis=1)
            center=(p*A[:,:,None]).sum(1)/A.sum(1)[:,None];vol6=np.linalg.det(p[:,1:]-p[:,:1]);assert vol6.min()>0
            radius=vol6/A.sum(1);keep=np.flatnonzero(np.all((center>=x.min(0))&(center<=x.max(0)),axis=1));candidates=[]
            for i in keep:
                pt=center[i];delta=np.maximum(np.maximum(low-pt,pt-high),0)
                d=np.sqrt(np.min(np.einsum('ij,ij->i',delta,delta)));ball=min(radius[i],d)-1.e-5
                if ball>0:candidates.append((ball,int(i),d))
            witnesses=[]
            for size,i,d in sorted(candidates,reverse=True)[:128]:
                loc=owner.point_location_prepared_indexed(tuple(Fraction.from_float(float(a)) for a in center[i]),prepared)['location']
                if loc=='inside':witnesses.append(dict(cell=i,center_m=center[i].tolist(),radius_mm=size*1000,bone_surface_distance_lower_bound_mm=d*1000))
            r['muscles'][s]=dict(checked_centers=min(len(candidates),128),inside_witnesses=witnesses,minimum_overlap_ml=sum(4*np.pi/3*(p['radius_mm']**3)/1000 for p in witnesses),largest_proven_depth_mm=max((p['bone_surface_distance_lower_bound_mm'] for p in witnesses),default=0))
    out.write_text(json.dumps(dict(receipt_sha256=sha(rc),bone_payload_sha256=bank[1],scope='Conservative sampled lower bounds at entrance. Upper bounds and total intersection volume unmeasured. The radius is bounded by the tet inradius and distance to the union of all bone triangle boxes with a 10 micrometre margin.',bones=result),indent=2)+'\n')
if __name__=='__main__':measure(*map(Path,sys.argv[1:]))
