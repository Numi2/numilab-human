"""Build source-identified closed shoulder material cells with exact boundary and pose checks."""
from pathlib import Path
import json,sys
import igl
import numpy as np
from igl.copyleft import cgal
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from repair_registered_surface import per_pose,state,volume
from prepare_winding_material import parents
from registration_stages import parts
from prepare_shoulder import read,tetrahedralize,MOVING
from compile_connected_joint import digest


def regularize(x,faces,w):
    y,fy,birth=cgal.mesh_boolean(x,faces,type_str='union')
    wy,ba,conflict=parents(y,fy,x,faces,birth,w)
    return y,fy,wy,birth,ba,conflict


def split_pinch(y,fy,w,alpha):
    nf,idx=igl.split_nonmanifold(fy)
    p=y[idx].copy()
    new=np.nonzero(np.bincount(idx)[idx]>1)[0]
    for i in new:
        adj=np.unique(nf[np.any(nf==i,axis=1)]);adj=adj[adj!=i]
        edges=p[adj]-p[i]
        # Replace each singular pole by one per physical fan, inside that fan.
        lengths=np.linalg.norm(edges,axis=1)
        axis=np.mean(edges/lengths[:,None],axis=0)
        p[i]+=alpha*lengths.min()*axis
    return p,nf,w[idx],new,idx


def frames(case):
    opening=read(case/'native-initial-bodies.json')['all_bodies']
    rc=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'))
    return [{r['body_index']:dict(quaternion_xyzw=r[k]['orientation_world_xyzw'],position_m=r[k]['position_world_m']) for r in opening} for k in ['rest','current']]+[{r['body_index']:r for r in read(rc)['accepted_body_poses']}]


def owner_repair(v,w,f,aff,b,poses,protect):
    adj=coo_matrix((np.ones(f.size*2),(np.r_[f.ravel(),np.roll(f,1,axis=1).ravel()],np.r_[np.roll(f,1,axis=1).ravel(),f.ravel()])),shape=(len(v),len(v))).tocsr()
    adj.data.fill(1);rows=np.asarray(adj.sum(1)).ravel()
    initial=w.copy();report=[]
    for step in range(36):
        hits=per_pose(v,w,f,aff,b,poses)
        report.append([len(h) for h in hits])
        if not any(report[-1]) or len(hits[0]):break
        local=np.unique(f[np.vstack(hits)])
        halo=np.asarray(adj[:,local].sum(1)).ravel()>0
        lam=np.where(halo,.08,0.);lam[local]=.45;lam[protect]=0.
        w=w+lam[:,None]*((adj@w)/rows[:,None]-w)
    return w,report,float(abs(w-initial).max())


def assemble(case,base,tool,out):
    out.mkdir(exist_ok=True)
    poses=frames(case);a=read(case/'launch.json')['argv'];source=Path(a[a.index('--soft-tissue-payload')+1])
    rows={}
    for sid in MOVING:
        raw,basis,owners,fc,aff=parts(source,sid);used=np.unique(fc)
        if sid in (75,85,87):
            ref=np.load(base/f'{sid}-atlas.npz');x=ref['source'].astype('f4').astype('f8');f=ref['faces'];w=ref['weights'];pins=ref['pins']
            y,fy,wy,birth,ba,conflict=regularize(x,f,w)
            fixed_points=x[pins]
            # Pin images of retained attachment-support vertices. Sites on the rig
            # remain owned by the NHTENDON payload. Track any removed support points.
            if not np.all(igl.is_vertex_manifold(fy)):
                yy,ff,ww,new,idx=split_pinch(y,fy,wy,.15)
            else:
                yy,ff,ww,new,idx=y,fy,wy,[],np.arange(len(y))
            dist,match=cKDTree(yy).query(fixed_points)
            protected=match[dist<1e-7];p_removed=int(np.count_nonzero(dist>=1e-7))
            wout,hist,change=owner_repair(yy,ww,ff,aff,owners,poses,protected)
        else:
            yy,sel,inv=np.unique(raw[used],axis=0,return_index=True,return_inverse=True)
            ff=inv[np.searchsorted(used,fc)];wout=basis[used[sel]]
            x,f=yy,ff;birth=np.arange(len(f));ba=np.broadcast_to(np.eye(3),(len(f),3,3));idx=np.arange(len(yy))
            conflict=change=0.;hist=[list(map(len,per_pose(yy,wout,ff,aff,owners,poses)))];p_removed=0;new=[]
        yy=yy.astype('f4').astype('f8');ff=np.ascontiguousarray(ff,np.int64)
        native=state(yy,wout,aff,owners,poses[-1]).astype('f4').astype('f8')
        bad=int((~igl.is_vertex_manifold(ff)).sum());ec=np.unique(np.sort(np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]]),axis=1),axis=0,return_counts=True)[1]
        line=dict(stages=hist,source_vertices=len(x),material_vertices=len(yy),edge_incidence_failures=int((ec!=2).sum()),vertex_link_failures=bad,
            weight_change=change,removed_anchor_vertices=p_removed,resolved_pinch_fans=len(new),source_weight_disagreement=conflict,
            volume_mm3=volume(yy,ff)*1e9,input_mm3=volume(x,f)*1e9,native_mm3=volume(native,ff)*1e9)
        np.savez_compressed(out/f'bank-{sid}.npz',x=yy,f=ff,w=wout,native=native,owners=owners,birth=birth,barycentric=ba,arrangement_vertex=idx)
        if not any(hist[-1]) and not bad and not line['edge_incidence_failures']:
            try:
                root=out/f'cells-{sid}';origin=native.mean(0)
                pts,t=tetrahedralize((native-origin)*1000,ff,tool,root)
                np.savez_compressed(root/'material.npz',x=pts/1000+origin,t=t,origin=origin)
                line['positive_native_cells']=len(t)
            except Exception as err:line['mesh_error']=str(err).replace(str(tool),'TetGen')[-150:]
        rows[sid]=line
        print(sid,'self',hist[0],hist[-1],'links',bad,'bound',line['edge_incidence_failures'],'missing anchors',p_removed,'tet',line.get('positive_native_cells',line.get('mesh_error')),flush=True)
        (out/'results.json').write_text(json.dumps(dict(rows=rows,source=digest(source),tetgen=digest(tool)),indent=2)+'\n')
if __name__=='__main__':assemble(*(Path(a).resolve()for a in sys.argv[1:]))
