"""Bind source quadrature to registered material cells and emit the existing native witness tables."""
import json, sys
from pathlib import Path
import numpy as np
from compose_registered_carriers import survivor_map
from register_material_surfaces import pinned, sha, tetgen_positions
from write_material_witnesses import write_role

def run(h,p66,job,flows,out):
    env=json.loads((job/'scene-material-p1/launch.json').read_bytes())['environment'];pre='NUMI_H706_FRACTIONAL_'
    def checked(path,digest):
        path=Path(path)
        if sha(path)!=digest:raise ValueError('Input identity differs: '+str(path))
        return path
    cells=np.loadtxt(pinned(env,'TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256'),skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    x=tetgen_positions(pinned(env,'TETGEN_NODE','TETGEN_NODE_SHA256')).astype('f4').astype('f8')
    labels=np.loadtxt(pinned(env,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    src=[]
    for what,cols,dtype in [('TETS',(3,4,5,6),'i4'),('NODES',(5,6,7),'f8')]:
        key=pre+'SOURCE330_'+what;src.append(np.loadtxt(checked(env[key],env[key+'_SHA256']),delimiter=',',skiprows=1,usecols=cols,dtype=dtype))
    W=survivor_map(np.load(h/'closed-native/source/material-source-indices.npz'));S=survivor_map(np.load(h/'closed-native/native/material-source-indices.npz'))
    mapping=np.full(len(W),-1,'i4');mapped=W>=0;mapping[mapped]=S[W[mapped]]
    radius=float(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M'])
    prepared=json.loads((flows/'result.json').read_bytes());result={};pins={};out.mkdir(exist_ok=True)
    for tag,mat in [('SKIN',15),('FAT',16)]:
        if tag not in prepared or not prepared[tag].get('flow_sha256'):raise ValueError('Missing complete '+tag+' allocation')
        k=tag.lower();sourcegeo=p66/'motion-repair-work/paired-p1'/(k+'-role-geometry.npz');old=np.load(sourcegeo)['combined_child_rows']
        ff=checked(flows/(tag+'-physical-flow.npz'),prepared[tag]['flow_sha256']);flow=dict(np.load(ff))
        key=pre+'CARRIERS_'+tag;packet=dict(np.load(checked(env[key+'_NPZ'],env[key+'_SHA256'])))
        key=pre+'P1_'+tag;p=dict(np.load(checked(env[key+'_NPZ'],env[key+'_SHA256'])))
        rows=np.flatnonzero(cells[:,4]==mat);idx=np.full(len(cells),-1,'i4');idx[rows]=np.arange(len(rows))
        center=x[cells[rows,:4]-1].mean(1)
        par=np.repeat(packet['global_source330_tet_rows'],np.diff(packet['parent_offsets']));points=np.einsum('ij,ijk->ik',packet['carrier_barycentric'],src[1][src[0][par]])
        region=labels[par,2];q=flow['carrier'];dest=flow['role_cell'];dist=np.linalg.norm(points[q]-center[dest],axis=1);within=dist<=radius
        prior=mapping[old[packet['carrier_target_child_rows']]];primary=np.full(len(par),-1,'i4');v=prior>=0;primary[v]=idx[prior[v]]
        old_distance=np.linalg.norm(points-center[np.maximum(primary,0)],axis=1)
        preserve=(primary>=0)&(old_distance<=radius);off=p['cell_admissible_region_offsets'];rids=p['cell_admissible_region_ids']
        edges_region=region[q]
        for rid in np.unique(region):
            valid=np.searchsorted(off[1:],np.flatnonzero(rids==rid),side='right');part=np.flatnonzero(edges_region==rid)
            within[part]&=np.isin(dest[part],valid)
            part=np.flatnonzero(region==rid);preserve[part]&=np.isin(primary[part],valid)
        primary[~preserve]=-1
        cuts=np.flatnonzero(~preserve[q]&within);order=cuts[np.lexsort((dest[cuts],dist[cuts],q[cuts]))]
        _,which=np.unique(q[order],return_index=True);primary[q[order[which]]]=dest[order[which]]
        if np.any(primary<0):raise ValueError('Missing current primary: '+str(np.flatnonzero(primary<0)[:20]))
        packet['carrier_target_child_rows']=primary.astype(packet['carrier_target_child_rows'].dtype)
        packet['carrier_target_distance_m']=np.linalg.norm(points-center[primary],axis=1)
        if 'regional_compensation_child_rows' in packet:
            rev=packet['regional_compensation_child_rows'];new=mapping[rev]
            if np.any(cells[new[new>=0],4]!=mat):raise ValueError('Compensation mapping changed material')
            packet['regional_compensation_child_rows']=np.sort(new[new>=0]).astype(rev.dtype)
        fp=out/(k+'-carriers.npz');np.savez_compressed(fp,**packet)
        witness=write_role(tag,mat,env,cells,x,labels,*src,packet,p,flow,radius,out/(k+'-witness.csv.gz'))
        result[tag]=dict(carriers=len(par),preserved_primary=int(preserve.sum()),rebound_primary=int((~preserve).sum()),maximum_primary_distance_m=float(max(packet['carrier_target_distance_m'])),maximum_flow_distance_m=float(dist.max()),out_of_radius_edges=int(np.sum(dist>radius)),witness=witness)
        for suffix,f,d in [('CARRIERS_'+tag,fp,sha(fp)),('CARRIER_WITNESS_'+tag,Path(witness['path']),witness['sha256'])]:
            key=pre+suffix;pins[key+('_NPZ' if suffix.startswith('CARRIERS_') else '_CSV')]=str(f);pins[key+'_SHA256']=d
        (out/'result.json').write_text(json.dumps(dict(roles=result,environment=pins),indent=2)+'\n');print(tag,result[tag],flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
