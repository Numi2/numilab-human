"""Compile the registered joint node map for the rebuilt, conservatively routed material."""
import csv
import itertools
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from register_material_surfaces import PREFIX,pinned,sha,vector,write_vector,tetgen_positions


class HistoricalTissue:
    def __init__(self,env):
        def source(k):
            path=Path(env['NUMI_H706_FRACTIONAL_SOURCE330_'+k])
            if sha(path)!=env['NUMI_H706_FRACTIONAL_SOURCE330_'+k+'_SHA256']:
                raise ValueError('Historical source differs')
            return path
        self.points=np.loadtxt(source('NODES'),delimiter=',',skiprows=1,usecols=(2,3,4))
        self.tet=np.loadtxt(source('TETS'),delimiter=',',skiprows=1,usecols=(3,4,5,6),dtype='i4')
        self.labels=np.loadtxt(pinned(env,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),skiprows=1,delimiter=',',dtype='i4',usecols=(1,2,3))
        if len(self.tet)!=len(self.labels):raise ValueError('Source domain differs')
        self.groups={}
        for i,g in enumerate(self.labels):self.groups.setdefault(tuple(g),[]).append(i)

    def project(self,x,key):
        ids=np.asarray(self.groups.get(tuple(key),[]),dtype=int)
        if not len(ids):raise ValueError(f'No source parent for material, donor and region {key}')
        corners=self.points[self.tet[ids]]
        best=np.full(len(ids),np.inf);w=np.zeros((len(ids),4))
        for size in range(1,5):
            for active in itertools.combinations(range(4),size):
                base=corners[:,active[0]]
                if size==1:
                    weights=np.ones((len(ids),1))
                else:
                    axes=(corners[:,active[1:]]-base[:,None]).transpose(0,2,1)
                    local=np.einsum('nij,nj->ni',np.linalg.pinv(axes),x-base)
                    weights=np.column_stack([1-local.sum(axis=1),local])
                valid=weights.min(axis=1)>=-1e-12
                weights=np.maximum(weights,0);weights/=weights.sum(axis=1)[:,None]
                point=np.einsum('nk,nkd->nd',weights,corners[:,active])
                dist=np.linalg.norm(point-x,axis=1);take=np.flatnonzero(valid & (dist<best))
                best[take]=dist[take];w[take]=0.;w[np.ix_(take,active)]=weights[take]
        i=int(best.argmin())
        return int(ids[i]),w[i],int(np.count_nonzero(abs(best-best[i])<=1e-10))


def compile_map(launch, material, layer_dir, parents, pose_file, dest):
    env=json.loads(launch.read_bytes())['environment']
    if sha(pose_file)!=json.loads((material/'receipt.json').read_bytes())['frame']:
        raise ValueError('Native donor snapshot differs from the certified material frame')
    node_map=pinned(env,'COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
    mesh=json.loads((material/'native-import.json').read_bytes())
    for f,k in [('mesh.node','node_sha256'),('mesh.ele','element_sha256'),('current-child.labels.csv','label_sha256')]:
        if sha(material/f)!=mesh[k]:raise ValueError('Changed imported material')
    x=tetgen_positions(material/'mesh.node')
    old=tetgen_positions(pinned(env,'TETGEN_NODE','TETGEN_NODE_SHA256'))
    D=np.loadtxt(material/'mesh.ele',skiprows=1,dtype='i4',usecols=(1,2,3,4,5))
    tet=D[:,:4]-1
    labels=np.loadtxt(material/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i4',usecols=(1,2,3))
    if not np.array_equal(labels[:,0],D[:,4]):raise ValueError('Changed incident material labels')
    cell_mass=np.fromfile(material/'cell-mass.f64',dtype='<f8')
    if sha(material/'cell-mass.f64')!=json.loads((material/'receipt.json').read_bytes())['cell_mass_sha256']:
        raise ValueError('Changed material mass')
    ancestry=np.load(material/'material-source-indices.npz')
    created=dict(zip(ancestry['new_steiner_native_indices'].tolist(),ancestry['new_steiner_source_indices'].tolist(),strict=True))
    retired=set(ancestry['retired_interior_vertex_ids'].tolist())
    if set(created)!=retired|set(range(len(old),len(x))):raise ValueError('Missing interior origin')
    order=np.argsort(tet.ravel(),kind='stable').astype('i4')
    offsets=np.r_[0,np.cumsum(np.bincount(tet.ravel(),minlength=len(x)))]
    cells=order//4
    old_tags=np.full((len(x),2),-1,dtype='i4')
    limits={'skin':0,'fat':0}
    template=None
    with node_map.open() as f:
        for row in csv.DictReader(f):
            idx=int(row['joint_node_id'])-1
            if idx not in created:old_tags[idx]=(int(row['geometric_donor_body']),int(row['anatomical_region_id']))
            if idx in retired and (row['role']!='interior' or row['source_parent_node_row'] not in ('-1','')):
                raise ValueError('Cannot retire a surface or historical parent node identity')
            if row['role']=='interior':template=row
            tag=row['source_partition'];limits[tag]=max(limits[tag],int(row['source_tetgen_node_id']))
    target_nodes=tet.ravel()[order]
    admissible=np.all(labels[cells,1:]==old_tags[target_nodes],axis=1)
    chosen=np.full(len(x),np.iinfo(np.int32).max,np.int32)
    np.minimum.at(chosen,target_nodes[admissible],order[admissible])
    unresolved=np.flatnonzero(chosen==np.iinfo(np.int32).max)
    missing = []
    for n in unresolved:
        if n in created: continue
        incident = cells[offsets[n]:offsets[n+1]]
        missing.append(dict(node=int(n), original=old_tags[n].tolist(),
                            incident_cells=incident.tolist(),
                            current=np.unique(labels[incident],axis=0).tolist()))
    for n in unresolved:
        inc=cells[offsets[n]:offsets[n+1]]
        if not len(inc):raise ValueError(f'Unoccupied material node {n}')
        candidates=labels[inc]
        if n not in created:
            # A recomputed integration cell can have a new geometric donor.
            # The rigid frame donor, immutable source parent and anatomical region
            # stay fixed. Only a genuine incident cell in that region can carry it.
            keep=np.flatnonzero(candidates[:,2]==old_tags[n,1])
            if not len(keep):raise ValueError(f'Existing node {n} has lost its anatomical region')
        else:
            if len(np.unique(candidates[:,0]))!=1:raise ValueError('Constructed interior unexpectedly joins tissues')
            keep=np.arange(len(inc))
        groups,inv=np.unique(candidates[keep],axis=0,return_inverse=True)
        group=np.argmax(np.bincount(inv,weights=cell_mass[inc[keep]]))
        local=keep[np.flatnonzero(inv==group)[0]]
        chosen[n]=order[offsets[n]+local]
    if np.any(tet.ravel()[chosen]!=np.arange(len(x))):raise ValueError('Unmatched operational tetrahedron')
    masks=np.zeros(len(x),dtype=np.int32)
    np.bitwise_or.at(masks,tet.ravel(),np.repeat(1<<labels[:,0],4))
    P=HistoricalTissue(env);pose=json.loads(pose_file.read_bytes())
    frames={b['body_index']:(np.asarray(b['current']['position_world_m'],dtype='f4').astype('f8'),np.reshape(b['current']['local_to_world_row_major_f32'],(3,3))) for b in pose['all_bodies']}
    binding={b['body_index']:i for i,b in enumerate(pose['bodies'])}
    # Copy the already native-validated boundary values to keep all reciprocal rows identical.
    layer_rows={}
    for k in 'OID':
        table=layer_dir/('registered-'+k+'.csv')
        if sha(table)!=json.loads((layer_dir/'registration.json').read_bytes())['layers'][k]['emitted']:raise ValueError('Changed surface')
        with table.open() as f:
            for r in csv.DictReader(f):
                layer_rows[k,int(r['vertex'])]=(vector(r,'l'),vector(r,'source_residual_','_m'))
    parent_source=parents/'node-parent-registration.csv'
    if sha(parent_source)!=json.loads((parents/'registration.json').read_bytes())['target']:raise ValueError('Changed parent table')
    parent_rows={}
    with parent_source.open() as f:
        for r in csv.DictReader(f):
            parent_rows[r['layer'],int(r['surface_vertex_row'])]=(vector(r,'parent_residual_','_m'),float(r['parent_registration_distance_m']))
    history=[]
    statistics={'max_new_source_projection_m':0.,'max_current_source_reconstruction_m':0.,'max_donor_reconstruction_m':0.}
    def emit(r,n,is_new):
        idx=chosen[n]//4;corner=chosen[n]%4;mat,donor,region=map(int,labels[idx]);point=x[n]
        if is_new:
            r=template.copy();tag='skin' if mat==15 else 'fat'
            limits[tag]+=1
            r.update(joint_node_id=n+1,source_partition=tag,source_tetgen_node_id=limits[tag],registered_I_source_vertex_row=-1,
                merge_peer_partition='',merge_peer_tetgen_node_id='',registered_surface_layer='',registered_surface_vertex_row=-1,
                role='interior',donor_body=donor,donor_binding=binding.get(donor,-1),source_domain='derived_volume_interior_no_surface_ancestry',
                source_mapping_kind='none',source_vertex_row=-1,source_face_row=-1,source_parent_node_row=-1,
                source_parent_geometric_donor_body=donor,source_parent_anatomical_region_id=region,
                parent_registration_kind='inferred_current_parent_tet_same_material_donor_region')
            parent,w,ties=P.project(point,(mat,donor,region))
            r.update(parent_tet_row=parent,parent_registration_tie_count=ties)
            for k in range(4):r['parent_bary'+str(k)]=format(w[k],'.17g')
            for k in range(3):r['source_bary'+str(k)]=0
            write_vector(r,'source_residual_',np.zeros(3),'_m')
            residual=point-w@P.points[P.tet[parent]];distance=float(np.linalg.norm(residual))
            if distance>float(env[PREFIX+'MAX_PARENT_REGISTRATION_DISTANCE_M']):raise ValueError(f'New interior {n} lacks a nearby source parent: {distance}')
            write_vector(r,'parent_current_residual_',residual,'_m')
            r['parent_registration_distance_m']=format(distance,'.17g')
            history.append((n,created[n],tag,limits[tag],parent,idx))
            statistics['max_new_source_projection_m']=max(statistics['max_new_source_projection_m'],distance)
        if int(r['incident_material_mask'])!=int(masks[n]) and not is_new:raise ValueError('Existing material interface changed')
        r.update(geometric_donor_body=donor,anatomical_region_id=region,incident_material_mask=int(masks[n]),operational_reference_tet_row=int(idx))
        for c in range(4):r['operational_reference_bary'+str(c)]=int(c==corner)
        write_vector(r,'operational_reference_residual_',np.zeros(3),'_m')
        r['operational_reference_distance_m']=0
        frame_owner=int(r['donor_body']);pos,R=frames[frame_owner]
        if r['role']!='interior':
            key=r['role'],int(r['registered_surface_vertex_row'])
            local,sres=layer_rows[key];pres,dist=parent_rows[key]
            write_vector(r,'source_residual_',sres,'_m');write_vector(r,'parent_current_residual_',pres,'_m')
            r['parent_registration_distance_m']=format(dist,'.17g')
        else:
            local=np.linalg.solve(R,point-pos)
            if not is_new and np.any(point!=old[n]):
                updated=vector(r,'parent_current_residual_','_m')+(point-old[n])
                write_vector(r,'parent_current_residual_',updated,'_m')
                r['parent_registration_distance_m']=format(np.linalg.norm(updated),'.17g')
        write_vector(r,'current_',point,'_m');write_vector(r,'donor_local_',local,'_m')
        row=int(r['parent_tet_row']);weights=np.asarray([float(r['parent_bary'+str(c)]) for c in range(4)])
        if tuple(P.labels[row,1:])!=(int(r['source_parent_geometric_donor_body']),int(r['source_parent_anatomical_region_id'])):raise ValueError('Unmatched historical tissue identity')
        e=np.linalg.norm(weights@P.points[P.tet[row]]+vector(r,'parent_current_residual_','_m')-point)
        f=np.linalg.norm(pos+R@local-point)
        if not np.isfinite(e+f) or e>1e-8 or f>1e-7:raise ValueError(f'Node {n} reconstruction differs: {e}, {f}')
        statistics['max_current_source_reconstruction_m']=max(e,statistics['max_current_source_reconstruction_m'])
        statistics['max_donor_reconstruction_m']=max(f,statistics['max_donor_reconstruction_m'])
        return r
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage=Path(tmp)/'volume';stage.mkdir()
        target=stage/'combined-node-map.csv'
        with node_map.open() as f,target.open('w') as out:
            reader=csv.DictReader(f);writer=csv.DictWriter(out,reader.fieldnames,lineterminator='\n');writer.writeheader()
            for r in reader:
                n=int(r['joint_node_id'])-1
                if n not in retired:writer.writerow(emit(r,n,False))
            for n in sorted(created):writer.writerow(emit(None,n,True))
        with (stage/'interior-source-rows.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['joint_node_row','generated_mesh_node_row','material_partition','partition_source_id','historical_parent_tet','incident_tet']);w.writerows(history)
        source=dict(launch=sha(launch),mesh=mesh['node_sha256'],cells=mesh['element_sha256'],native_surface_table=sha(layer_dir/'registration.json'),parents=sha(parent_source),source_combined=sha(node_map),pose=sha(pose_file))
        (stage/'registration.json').write_text(json.dumps(dict(source=source,volume_nodes=len(x),retained=len(x)-len(created),new_interior=len(created),
            replaced_interior_slots=len(retired),reselected_incident_donors=missing,combined_map_sha256=sha(target),**statistics),indent=2)+'\n')
        if dest.exists():raise FileExistsError(dest)
        stage.rename(dest)


if __name__=='__main__':compile_map(*(Path(v).resolve() for v in sys.argv[1:]))
