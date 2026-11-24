"""Map reference and bed-pose volume resolution to the actual tissue regions."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from condition_motion_witness import altitude
from compile_connected_joint import digest

def run(work):
    v=work/'witness-material/volume'
    pts=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    x=pts(v/'mesh.node');y=pts(v/'current.node')
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    tet=raw[:,:4]-1
    labels=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')
    mass=np.fromfile(v/'cell-mass.f64','<f8')
    h=np.empty(len(tet))
    for a in range(0,len(tet),32768):
        ids=slice(a,a+32768);h[ids]=np.minimum(altitude(x,tet[ids]),altitude(y,tet[ids]))*1e6
    # If the altitude of a fixed boundary triangle is small, every incident
    # tetrahedron has a minimum altitude at most that small in the same pose.
    def keys(f):
        return np.ascontiguousarray(np.sort(f,axis=1),dtype='<i4').view('V12').ravel()
    def triangle_altitude(p,f):
        a=p[f];u=a[:,1]-a[:,0];v=a[:,2]-a[:,0]
        return np.linalg.norm(np.cross(u,v),axis=1)/np.maximum.reduce([np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1),np.linalg.norm(v-u,axis=1)])*1e6
    fixed={}
    for k,face in np.load(v/'material-walls.npz').items():
        hf=np.minimum(triangle_altitude(x,face),triangle_altitude(y,face))
        fixed[k]=keys(face[hf<1.])
    report=dict(elements=digest(v/'mesh.ele'),metric='Smallest vertex-to-opposite-face altitude of each tetrahedron, minimum of reference and supported starting pose. Screening thresholds are descriptive and do not establish dynamic failure.',roles={})
    for name,mat in [('skin',15),('fat',16)]:
        use=labels[:,1]==mat;counts={}
        for threshold in [1.,5.,10.,25.]:
            sel=use&(h<threshold);idx=np.flatnonzero(sel);b,n=np.unique(labels[sel,3],return_counts=True);order=np.argsort(-n)
            # Cell stars are coupled whenever cells share a mechanical node.
            if len(idx):
                nodes,inv=np.unique(tet[idx],return_inverse=True)
                ii=np.repeat(np.arange(len(idx)),4);jj=len(idx)+inv.ravel()
                graph=coo_matrix((np.ones(len(ii),'u1'),(ii,jj)),shape=(len(idx)+len(nodes),len(idx)+len(nodes))).tocsr()
                groups,cc=connected_components(graph,directed=False);nper=np.bincount(cc[:len(idx)])
            else:groups=0;nper=np.zeros(1,int)
            counts[str(threshold)]=dict(cells=int(sel.sum()),mass_kg=float(mass[sel].sum()),connected_stars=int(groups),largest_star_cells=int(nper.max()),source_regions=len(b),most_affected=[{'anatomical_region':int(b[j]),'cells':int(n[j])}for j in order[:5]])
        ids=np.flatnonzero(use&(h<1.));tt=tet[ids];corners=tt[:,((1,2,3),(0,2,3),(0,1,3),(0,1,2))].reshape(-1,3)
        q=keys(corners)
        active={r:np.isin(q,vals).reshape(-1,4).any(1) for r,vals in fixed.items()}
        forced=np.logical_or.reduce(list(active.values()))
        report['roles'][name]=dict(submicron_cells_incident_to_submicron_registered_face=int(forced.sum()),cells_by_constraining_wall={r:int(a.sum()) for r,a in active.items()},total=int(use.sum()),minimum_um=float(h[use].min()),counts_below_um=counts)
    (work/'witness-material/resolution.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run(Path(__file__).resolve().parent)
