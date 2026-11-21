"""Evaluate a shared anatomical registration field on the captured layers and their volumetric rest mesh."""
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from shoulder_inputs import digest,old

LAYERS=(1,0xfffffd11,0xfffffd12)
BONES=(8,9,10,11)
def inputs(scene,loader,receipt,bone):
    cfg=json.loads((scene/'launch.json').read_text())
    os.environ['NUMI_HUMAN_CAPTURED_TIME_ZERO_RECEIPT']=str(receipt)
    sys.argv[1]=str(scene)
    state=runpy.run_path(str(loader))
    banks=state['banks'];banks[51007,1]=(state['skin'],state['skin_faces'])

    old.audit.ag.RIGID=Path(cfg['argv'][1])
    meshes=[]
    for file in (Path(cfg['argv'][3]),bone):
        old.audit.ag.BONES=file
        _,raw,_,_,_,_,_,cat,_,vo,io=old.audit.ag.read_payloads()
        meshes.append(old.load_local_meshes(raw,cat,vo,io))
    surfaces={}
    for b in BONES:
        pose=state['poses'][cat[b]['body']]
        rr=Rotation.from_quat(pose['quaternion_xyzw'])
        pair=[(rr.apply(mesh[b][0])+pose['position_m']).astype('f4') for mesh in meshes]
        assert np.array_equal(meshes[0][b][1],meshes[1][b][1])
        xc=banks[51004,b][0]
        assert cKDTree(xc).query(pair[0])[0].max()<2e-6 and cKDTree(pair[0]).query(xc)[0].max()<2e-6
        surfaces[b]=(*pair,meshes[0][b][1])
    keys='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_'
    def input_path(name):
        p=Path(cfg['environment'][keys+name])
        assert digest(p)==cfg['environment'][keys+name+'_SHA256']
        return p
    nodes=input_path('NODE'); elements=input_path('ELEMENTS')
    pts=np.loadtxt(nodes,skiprows=1,usecols=(1,2,3),dtype=np.float32)
    return state,banks,surfaces,pts,{'pose':digest(receipt),'base_pack':state['zero']['base_pack_file_sha256'],'bones':digest(bone),'nodes':digest(nodes),'elements':digest(elements)}

def measure(x,faces,normal,offset):
    gap=x.astype(float)@normal-offset
    witness=int(gap.argmin())
    return dict(minimum_gap_m=float(gap[witness]),vertex=witness,point_world_m=x[witness].tolist(),
                penetrating_vertices=int((gap<0).sum()),
                penetrating_faces=int(np.any(gap[faces]<0,axis=1).sum()))

def run(scene,loader,receipt,bone,sources,source_manifest,reference,out):
    state,banks,bs,pts,pins=inputs(scene,loader,receipt,bone)
    logged=re.findall(r'^resting_fem_skin_bed_source=enabled .*?normal=([^ ]+) plane_offset_m=([^ ]+)',(scene/'stdout.log').read_text(),flags=re.M)
    assert len(set(logged))==1,'Missing or inconsistent active native FEM bed plane'
    normal=np.asarray(list(map(float,logged[0][0].split(','))));offset=float(logged[0][1])
    assert abs(np.linalg.norm(normal)-1)<1e-5
    g=np.load(sources);meta=json.loads(source_manifest.read_text())
    for row in meta['bones']:
        assert hashlib.sha256(g[row['body']+'_world'].tobytes()).hexdigest()==row['sha256_world_float64']
    cfg=json.loads((scene/'launch.json').read_text());parent=Path(cfg['argv'][3])
    doc=json.loads(parent.with_suffix('.manifest.json').read_text())
    assert doc['payload']['sha256']==digest(parent)
    tree=cKDTree(pts);layers=[]
    for s in LAYERS:
        x,f=banks[51007,s]
        distance=tree.query(x)[0]
        assert distance.max()==0.,'Physical volume and captured surface coordinates differ'
        layers.append(dict(stable=s,vertices=len(x),max_node_lookup_distance_m=float(distance.max()),**measure(x,f,normal,offset)))
    rows=[]
    for b,(old_points,new_points,faces) in bs.items():
        row=doc['source']['anchors'][b-1];name=row['myosim_body'];body=row['core_body_index']
        q=state['poses'][body]
        points=(Rotation.from_quat(q['quaternion_xyzw']).apply(g[name+'_local'])+q['position_m']).astype('f4')
        # A second reconstruction at the imported default pose pins the donor
        # local frame before it is applied at the current supported pose.
        ref=reference[row['core_body_index']]
        world=Rotation.from_quat(ref['default_inertial_quaternion_world_xyzw']).apply(g[name+'_local'])+ref['default_com_position_world_m']
        assert abs(world-g[name+'_world']).max()<1e-6
        rows.append(dict(stable=b,name=name,parent=measure(old_points,faces,normal,offset),
            candidate=measure(new_points,faces,normal,offset),
            physical_source=measure(points,g[name+'_faces'],normal,offset)))
    result=dict(scope='Native initialization. Necessary bed half-space and physical-layer correspondence before tissue transfer.',
        normal=normal.tolist(),offset_m=offset,inputs=pins,volume_nodes=len(pts),source_archive=meta['source_archive_sha256'],
        physical_layers=layers,bones=rows,eligible_for_tissue_transport=all(r['candidate']['penetrating_vertices']==0 for r in rows))
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('bed feasibility',result['eligible_for_tissue_transport'],[(r['name'],r['candidate']['minimum_gap_m']) for r in rows],flush=True)
    return result['eligible_for_tissue_transport']

if __name__=='__main__':
    args=list(map(Path,sys.argv[1:]))
    scene=args[0];cfg=json.loads((scene/'launch.json').read_text());bp=Path(cfg['argv'][3])
    mp=json.loads(bp.with_suffix('.manifest.json').read_text())['source']['runtime_reference']['manifest']
    path=Path(mp['file']);assert digest(path)==mp['sha256']
    reference={r['core_body_index']:r for r in json.loads(path.read_text())['core_tree']['source_body_records']}
    raise SystemExit(0 if run(*args[:-1],reference,args[-1]) else 1)
