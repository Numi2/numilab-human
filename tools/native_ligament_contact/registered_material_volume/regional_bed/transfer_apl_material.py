"""Transfer identical source-material stations across compiler revisions after proving identity."""
import json,runpy,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from fit_route_binding import asset,read,check_shapes
from measure_forearm_registration import evaluate,load_frame
from registration_stages import parts

def run(reader,control,donor,out):
    out.mkdir()
    config=read(control/'launch.json')
    donorcfg=read(donor/'launch.json')
    src=asset(config,'--soft-tissue-payload')
    previous=asset(donorcfg,'--soft-tissue-payload')
    desc=previous.with_suffix('.manifest.json')
    manifest=read(desc)
    if manifest['payload']['sha256']!=digest(previous):raise ValueError('Unmatched donor payload')
    reg={i['stable_id']:i for i in manifest['source']['surfaces']}
    weights={};inputs={};guides={}
    for sid in (147,148):
        x,w,b,f,aff=parts(src,sid)
        old,v,bb,ff,af=parts(previous,sid)
        if not(np.array_equal(old,x) and np.array_equal(f,ff) and b==bb and all(np.array_equal(i,j) for l,r in zip(aff,af) for i,j in zip(l,r))):
            raise ValueError('Source material, connectivity or donor frame changed')
        weights[str(sid)]=v.astype('<f4')
        inputs[sid]=(x,w,b,f,aff)
        guides[sid]=reg[sid]['route_binding']['guides']
    np.savez_compressed(out/'bindings.npz',**weights)
    api=runpy.run_path(str(reader))
    rc,=(control/'native-traces/accepted-geometry').glob('*.receipt.json')
    results={}
    for initial in (True,False):
        meshes,sha=load_frame(api,control,rc,initial)
        capture=read(rc)
        poses={p['body_index']:p for p in (capture['initial_anatomical_registration']['body_poses'] if initial else capture['accepted_body_poses'])}
        if initial:
            for p in read(control/'native-initial-bodies.json')['all_bodies']:
                poses.setdefault(p['body_index'],dict(position_m=p['current']['position_world_m'],quaternion_xyzw=p['current']['orientation_world_xyzw']))
        records={};surfaces=[];labels=[];qualities=[]
        for sid,(x,w,b,f,aff) in inputs.items():
            for mode,ww in [('control',w),('routed',weights[str(sid)])]:
                geom,quality=check_shapes(x,ww,b,f,aff,poses,meshes[51005,sid] if mode=='control' else None,mode)
                surfaces.append(geom);labels.append((sid,mode));qualities.append(quality)
        for (sid,mode),q,m in zip(labels,qualities,evaluate(meshes[51007,1],surfaces)):
            records.setdefault(sid,{})[mode]=dict(**q,**m)
        tag='release' if initial else 'accepted'
        results[tag]=dict(pack=sha,rows=records)
        print(tag,[(sid,row['routed']['self_pairs'],row['routed']['max_vertex_mm'],row['routed']['volume_cm3'],row['routed']['source_scaled_range_cm3'])for sid,row in records.items()],flush=True)
    result=dict(launch=digest(control/'launch.json'),tissue=digest(src),muscles=digest(Path(config['argv'][2])),tendons=digest(asset(config,'--tendon-payload').with_suffix('.manifest.json')),
                candidates=digest(out/'bindings.npz'),guides=guides,poses=results,source_material_stations=dict(registered_source=digest(previous),source_manifest=digest(desc),
                reference_route_sha256=manifest['source']['forearm_binding_parent']['route_program_sha256'],original_material_geometry_connectivity_and_donor_frames_match=True))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
