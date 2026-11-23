"""Locate source muscle route sites against the installed loaded exterior at the same native pose."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from registered_forearms import native_sites
from run_bed_work import digest


class TriangleDistance:
    def __init__(self,v,f):
        self.tri=np.asarray(v,dtype=float)[f]
        center=self.tri.mean(1);self.tree=cKDTree(center)
        self.radius=float(np.linalg.norm(self.tri-center[:,None],axis=2).max())
        self.lo=self.tri.min(1);self.hi=self.tri.max(1)

    def nearest(self,p):
        _,j=self.tree.query(p)
        upper=float(np.linalg.norm(self.tri[j]-p,axis=1).min())
        tri=self.tri[self.tree.query_ball_point(p,upper+self.radius+1e-10)]
        edges=np.roll(tri,-1,axis=1)-tri;diff=p-tri
        normalize=lambda x:np.einsum('...j,...j->...',x,x)
        length=normalize(edges)
        if np.any(length<=0):raise ValueError('Degenerate tissue boundary')
        u=np.clip(np.einsum('...j,...j->...',diff,edges)/length,0,1)
        edge_d2=normalize(diff-u[:,:,None]*edges).min(1)
        normal=np.cross(edges[:,0],-edges[:,2]);nn=normalize(normal)
        if np.any(nn<=0):raise ValueError('Singular tissue face')
        side=np.einsum('ijk,ik->ij',np.cross(edges,diff),normal)
        plane=np.einsum('ij,ij->i',diff[:,0],normal)**2/nn
        d2=np.where(np.all(side>=0,axis=1),np.minimum(edge_d2,plane),edge_d2)
        return float(np.sqrt(d2.min()))


class Surface(TriangleDistance):
    def __init__(self,v,f):
        super().__init__(v,f)
        edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        edge,count=np.unique(edges,axis=0,return_counts=True)
        if max(count)>2:raise ValueError('Nonmanifold outer boundary')
        self.boundary=edge[count==1]
        holes=v[np.unique(self.boundary)]
        self.excluded=(holes.min(0),holes.max(0)) if len(holes) else None

    def query(self,p):
        votes=[];eps=1e-9
        for axis,axes in ((0,(1,2)),(1,(2,0)),(2,(0,1))):
            a,b=axes
            if self.excluded is not None:
                lo,hi=self.excluded
                if lo[a]-eps<=p[a]<=hi[a]+eps and lo[b]-eps<=p[b]<=hi[b]+eps:continue
            tri=self.tri[(self.lo[:,a]<=p[a]+eps)&(p[a]<=self.hi[:,a]+eps)&(self.lo[:,b]<=p[b]+eps)&(p[b]<=self.hi[:,b]+eps)]
            edges=tri[:,1:]-tri[:,:1];rx=p[a]-tri[:,0,a];ry=p[b]-tri[:,0,b]
            xa,xb=edges[:,:,a].T;ya,yb=edges[:,:,b].T;area=xa*yb-xb*ya
            if np.any(abs(area)<1e-16):continue
            u=(rx*yb-ry*xb)/area;v=(ry*xa-rx*ya)/area
            bary=np.min(np.c_[u,v,1-u-v],axis=1)
            if np.any(abs(bary)<eps):continue
            use=bary>0
            hits=tri[:,0,axis]+u*edges[:,0,axis]+v*edges[:,1,axis]-p[axis]
            if np.any(abs(hits[use])<eps):continue
            pos=int(np.count_nonzero(hits[use]>0));neg=int(np.count_nonzero(hits[use]<0))
            if pos%2!=neg%2:raise ValueError('Inconsistent parity outside aperture projection')
            votes.append(pos%2)
        if votes and len(set(votes))!=1:raise ValueError('Independent ray directions disagree')
        label=('inside' if votes[0] else 'outside') if votes else 'indeterminate'
        return dict(side=label,qualified_axis_count=len(votes),distance_m=self.nearest(p))


def audit(root):
    def read(p):return json.loads(p.read_text())
    records=read(root/'pose-one/terminal.receipt.json')
    poses={r['body_index']:r for r in records['initial_anatomical_registration']['body_poses']}
    cfg=read(root/'pose-one/launch.json');args=cfg['argv'];paths=root/'material-source/corrected'
    def pin(p):
        p=Path(p)
        if digest(p)!=cfg['pins'][str(p)]:raise ValueError('Source changed: '+p.name)
        return p
    tissue=pin(args[args.index('--soft-tissue-payload')+1]);meta=read(pin(tissue.with_suffix('.manifest.json')))
    myo=pin(args[2]);raw=myo.read_bytes()
    ten=pin(args[args.index('--tendon-payload')+1]);tmeta=read(ten.with_suffix('.manifest.json'))
    if (tmeta['payload']['sha256']!=digest(ten) or tmeta['source']['myosim_muscle_payload_sha256']!=digest(myo)
        or meta['payload']['sha256']!=digest(tissue)):raise ValueError('Source registration differs')
    tags={(k['muscle'],k['endpoint']):k for k in tmeta['endpoints']}
    proof=read(paths/'connected-columns.json');surface=paths/'outer-boundaries.npz'
    if digest(surface)!=proof['outer_boundaries'] or proof['datum_pose']!=digest(root/'pose-one/terminal.receipt.json'):raise ValueError('Different material frame')
    with np.load(surface) as z:s=Surface(z['loaded_xyz'],z['triangles'])
    catalog={r['stable_id']:r for r in meta['source']['surfaces'] if r['layer']=='muscle'}
    banks=read(root/'material-source/muscle-witnesses.json')['banks']['dense_loaded']
    ids={};mapped=[];no_route=[]
    for bank in banks:
        if not bank['surface_pairs']:continue
        sid=bank['id'];surf=catalog[sid];routes=surf.get('matched_muscles',[])
        if not routes:no_route.append(sid);continue
        for rt in routes:
            name=rt['name'];points=native_sites(raw,name,rt['source_actuator_index'],rt['source_route_node_count'],tags)
            for i,site in enumerate(points):
                idx=site['site_index']
                if idx not in ids:
                    bone=poses[site['body']];p=Rotation.from_quat(bone['quaternion_xyzw']).apply(site['local_m'])+bone['position_m']
                    ids[idx]=dict(body=site['body'],world_m=p.tolist(),**s.query(p))
                mapped.append(dict(bank=sid,route=name,site=idx,kind='origin' if i==0 else 'insertion' if i+1==len(points) else 'intermediate'))
    stats={}
    for kind in ('origin','insertion','intermediate'):
        pts={r['site'] for r in mapped if r['kind']==kind}
        stats[kind]={side:sum(ids[i]['side']==side for i in pts) for side in ('inside','outside','indeterminate')}
    return dict(opening=proof['datum_pose'],boundary=digest(surface),myosim=digest(myo),tendon=digest(ten),open_edges=len(s.boundary),sites=ids,
                route_roles=mapped,unique_sites_by_role=stats,unmatched_surface_ids=no_route,
                scope='Route sites on the registered dense loaded exterior at initialization. Two-sided parity discards rays within the projected bounds of all openings and ambiguous edge hits. Distances are to the nearest triangle. This test does not certify paths between sites.')
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    (root/'material-source/route-clearance.json').write_text(json.dumps(audit(root),indent=2,allow_nan=False)+'\n')
