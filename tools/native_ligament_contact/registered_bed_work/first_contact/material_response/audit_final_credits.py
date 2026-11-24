"""Check the post-balance edges for every newly constructed cell against the source quadrature."""
import sys,json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def run(out):
    read=lambda p:json.loads(p.read_bytes())
    cfg=read(out/'scene-material-p1/launch.json')['environment']
    src=read(out/'material-p1/current-paired-volume-state.json')['source330_identity']
    def source(k,cols,dtype):
        info=src['source330_'+k];p=Path(info['path'])
        if digest(p)!=info['sha256']:raise ValueError('Source changed')
        return np.loadtxt(p,delimiter=',',skiprows=1,usecols=cols,dtype=dtype)
    src_nodes=source('nodes',(5,6,7),'f8');src_cells=source('tets',(3,4,5,6),'i4');regions=source('labels',3,'i4')
    v=out/'volume';z=np.load(v/'overlay.npz');new=np.unique(z['target'])
    e=np.loadtxt(v/'mesh.ele',usecols=(1,2,3,4,5),skiprows=1,dtype='i4')
    x=np.loadtxt(v/'mesh.node',usecols=(1,2,3),skiprows=1).astype('f4').astype('f8')
    proof={};P='NUMI_H706_FRACTIONAL_';limit=float(cfg['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M'])
    for role,m in [('SKIN',15),('FAT',16)]:
        rr=np.flatnonzero(e[:,4]==m);f=np.load(out/'flows'/(role+'-physical-flow.npz'));c=f['role_cell'];s=np.isin(rr[c],new)
        p=np.load(out/'packets'/(role.lower()+'-carriers.npz'));par=np.repeat(p['global_source330_tet_rows'],np.diff(p['parent_offsets']))
        q=f['carrier'][s];target=c[s];centers=x[e[rr[target],:4]-1].mean(1)
        pts=np.einsum('ij,ijk->ik',p['carrier_barycentric'][q],src_nodes[src_cells[par[q]]]);d=np.linalg.norm(pts-centers,axis=1)
        g=np.load(cfg[P+'P1_'+role+'_NPZ']);o=g['cell_admissible_region_offsets'];ids=g['cell_admissible_region_ids']
        bad=sum(regions[par[a]] not in ids[o[b]:o[b+1]] for a,b in zip(q,target))
        proof[role]=dict(new_receiver_edges=int(s.sum()),maximum_new_distance_m=float(max(d)),radius_m=limit,ineligible_regions=int(bad),outside_radius=int((d>limit).sum()))
        if bad or np.any(d>limit):raise ValueError(str(proof))
    (out/'final-credits.json').write_text(json.dumps(proof,indent=2)+'\n');print(proof)
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
