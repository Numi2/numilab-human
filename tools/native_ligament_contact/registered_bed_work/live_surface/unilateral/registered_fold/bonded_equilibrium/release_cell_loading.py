"""Compare the opposing native volume limits to the same cells in the material and accepted configurations."""
import csv,json,re,sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import installed_topology
from witness import geometry

def read(p):
    return json.loads(p.read_text())

def compare(case):
    base=case/'held-volume';run=base/'six';frame=base/'three'
    old=read(case/'witness.json');long=read(base/'six-summary.json');f=read(run/'failure.json')
    if long['prefix_identical']!=3 or long['accepted']!=3 or not long['rollback'] or f['matter_diagnostics'][3]!=4:
        raise ValueError('Expected upper volume witness after matched third step')
    log=(run/'stdout.log').read_text()
    def rows(pattern,expected):
        r=re.findall(pattern,log,re.M)
        if len(r)!=expected:raise ValueError('Unmatched native object inventory')
        return np.asarray(r,dtype=int)
    fascia=rows(r'^resting_fascia_object=source_prepared object=\d+ .*?nodes=(\d+) tetrahedra=(\d+)',12)
    acl=rows(r'^resting_inferred_acl=admitted .*?nodes=(\d+) tetrahedra=(\d+)',2)
    outer,wall=rows(r'^resting_aortic_wall_outer_contact=accepted .*?outer_ring_rigid_query_nodes=(\d+) .*?retained_wall_faces=(\d+)',1)[0]
    offset=fascia.sum(axis=0)+acl.sum(axis=0)+np.array([2*outer,3*wall])
    if offset[0]!=rows(r'\bfirst_global_outer_fem_node=(\d+)\b',1)[0]:raise ValueError('Wrong compiled prefix')
    j=read(run/'launch.json');e=j['environment']
    X,Y,base_cells,base_mat,pins=inputs(run)
    cells,material=installed_topology(base_cells,base_mat,e)
    if pins['installed_pose_sha256']!=read(base/'verified-mass.json')['loaded_pose']:raise ValueError('Different mass registration')
    z=frame/'physical-accepted.node';meta=read(frame/'verified-endpoint.json')
    if sha(z)!=meta['physical_nodes_sha256'] or meta['accepted_steps']!=3:raise ValueError('Missing accepted physical endpoint')
    Z=np.loadtxt(z,skiprows=1,usecols=(1,2,3));p='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def checked(k):
        path=Path(e[p+k]);key=k.replace('_CSV','')+'_SHA256'
        if sha(path)!=e[p+key]:raise ValueError('Changed '+k)
        return path
    labels=checked('CHILD_LABELS')
    node_map=checked('COMBINED_NODE_MAP_CSV')
    with np.load(e['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ']) as c:keep=c['retained_input_cell_rows']
    records=[]
    for label,native in [('compression',old['native_cell']),('expansion',int(f['matter_diagnostics'][2]))]:
        i=native-int(offset[1])
        if not 0<=i<len(keep):raise ValueError('Witness requires composite replacement ancestry')
        src=int(keep[i]);n=cells[i]
        if not np.array_equal(n,base_cells[src]) or (label=='compression' and list(n)!=old['nodes']):raise ValueError('Different physical cell')
        g={s:geometry(coords[n].astype(float)) for s,coords in [('reference',X),('start',Y),('step3',Z)]}
        def edges(arr):
            coords=arr[n].astype(float)
            return coords[1:]-coords[:1]
        stretches=np.linalg.svd(np.linalg.solve(edges(X),edges(Z)),compute_uv=False)
        records.append(dict(limit=label,native_cell=native,source_cell=src,material=int(material[i]),nodes=list(map(int,n)),
            center_material_m=X[n].astype(float).mean(axis=0).tolist(),minimum_material_altitude_um=1e6*min(g['reference']['altitudes_m']),
            condition_edge_matrix=g['reference']['edge_condition'],start_J=g['start']['volume_m3']/g['reference']['volume_m3'],accepted_J=g['step3']['volume_m3']/g['reference']['volume_m3'],
            accepted_over_start=g['step3']['volume_m3']/g['start']['volume_m3'],accepted_principal_stretches=stretches.tolist()))
    owners={r['source_cell']:r for r in records}
    with labels.open() as file:
        for i,row in enumerate(csv.DictReader(file)):
            if i in owners:
                if int(row['child_tet_row'])!=i or int(row['material_index'])!=owners[i]['material']:raise ValueError('Wrong tissue identity')
                owners[i]['geometric_donor_body']=int(row['geometric_donor_body']);owners[i]['anatomical_region_id']=int(row['anatomical_region_id'])
    needed={n for r in records for n in r['nodes']};roles={}
    with node_map.open() as file:
        for row in csv.DictReader(file):
            i=int(row['joint_node_id'])-1
            if i in needed:
                layer=row['registered_surface_layer'] or 'interior'
                if roles.setdefault(i,layer)!=layer:raise ValueError('Ambiguous interface')
    for r in records:r['node_layers']=[roles[n] for n in r['nodes']]
    return dict(geometry=pins,accepted_geometry_sha256=sha(z),failed_launch_sha256=sha(run/'launch.json'),global_element_offset=int(offset[1]),
        before_failure=records,witness_centers_separation_m=float(np.linalg.norm(np.subtract(*[r['center_material_m'] for r in records]))),
        failed_trial_J=f['matter_diagnostics'][0],current_newton_J=f['matter_diagnostics'][1],newton_iteration=f['matter_failing_index'],
        trial_reason='upper material determinant bound',failed_step=4)

if __name__=='__main__':print(json.dumps(compare(Path(sys.argv[1]).resolve()),indent=2,allow_nan=False))
