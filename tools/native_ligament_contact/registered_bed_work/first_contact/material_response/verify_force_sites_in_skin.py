"""Classify all registered muscle route sites against the physical skin at native release."""
from collections import Counter
from pathlib import Path
import json
import struct
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from recount_registered_crossings import digest, read


def traction_nodes(tendon, metadata, pose):
    fields = struct.unpack_from('<8s10I32s32s32s',tendon)
    start = 144 + 64*fields[5]
    points, rows, indices = [], [], set()
    for i, row in enumerate(metadata['endpoints']):
        ep = struct.unpack_from('<8I8f',tendon,144+64*i)
        mode, index = ep[5:7]
        if mode == 0:
            continue
        if mode not in (2,3) or index in indices or index>=fields[6]:
            raise ValueError('Invalid traction patch')
        indices.add(index)
        blob = struct.unpack_from('<4I68f',tendon,start+288*index)
        if (blob[0],blob[1],blob[3]) != (ep[4],ep[7],4):
            raise ValueError('Attachment and physical carrier disagree')
        local=np.array(blob[4:20]).reshape(4,4)[:,:3]
        maps=np.array(blob[20:68]).reshape(4,3,4)[:,:,:3]
        if not np.array_equal(local,np.asarray(row['surface']['node_local_points_m'],dtype='f4')):
            raise ValueError('Traction node coordinate differs from the native asset')
        # Unit-load responses in the endpoint body frame.
        force=float(np.max(np.abs(maps.sum(axis=0)-np.eye(3))))
        arms=local - np.asarray(ep[8:11])
        moment=float(np.max(np.abs(np.cross(arms[:,None,:], maps.transpose(0,2,1)).sum(axis=0))))
        state=pose[ep[4]]
        pts=Rotation.from_quat(state['orientation_world_xyzw']).apply(local)+state['position_world_m']
        for k,q in enumerate(pts):
            points.append(q)
            rows.append(dict(muscle=row['muscle'],role=row['endpoint'],body=ep[4],mode=mode,
                             bone=ep[7],endpoint=i,node=k,unit_force_residual=force,unit_moment_residual_m=moment))
    if indices!=set(range(fields[6])):
        raise ValueError('Incomplete native traction patch accounting')
    return points,rows


def audit(reference, trial, export, clearance, lib, output):
    from measure_forearm_enclosure import Enclosure, test
    test(lib)
    launch = read(trial/'launch.json')
    args = launch['argv']
    core = Path(args[2])
    tendon = Path(args[args.index('--tendon-payload')+1])
    if any(digest(p) != launch['pins'][str(p)] for p in (core,tendon)):
        raise ValueError('Native payload differs from the control')
    m, t = core.read_bytes(), tendon.read_bytes()
    mh = struct.unpack_from('<8s9I32s',m)
    _, abi, _, nm, ns, nw, nr, _, _, _, _ = mh
    th = struct.unpack_from('<8s10I32s32s32s',t)
    if th[0:2] != (b'NHTEND3\x00',3) or th[3:6] != (nm,ns,2*nm) or th[-2].hex() != digest(core):
        raise ValueError('Unsupported registered tendon routes')
    route = 76+16*ns+64*nw
    muscle = route+16*nr
    if mh[0]!=b'NHMYO2\x00\x00' or abi!=2 or len(m)!=muscle+196*nm or len(t)!=144+128*nm+288*th[6]:
        raise ValueError('Unexpected muscle or tendon layout')
    tm = read(tendon.with_suffix('.manifest.json'))
    if tm['payload']['sha256']!=digest(tendon):
        raise ValueError('Stale muscle identities')
    ep = {}
    for i, tag in enumerate(tm['endpoints']):
        fields = struct.unpack_from('<8I8f',t,144+64*i)
        key = (tag['muscle_index'],int(tag['endpoint']=='insertion'))
        if key!=fields[:2] or fields[2:5]!=(tag['route_node_index'],tag['source_site_index'],tag['body_index']) or not np.array_equal(np.asarray(tag['resolved_local_point_m'],dtype='f4'),fields[8:11]):
            raise ValueError('Tendon binary/manifest discrepancy')
        if key in ep:
            raise ValueError('Duplicate force endpoint')
        ep[key] = (fields,tag['muscle'])
    if len(ep)!=2*nm:
        raise ValueError('Incomplete mechanical endpoints')
    posefile = trial/'native-initial-bodies.json'
    cert = read(clearance)
    if digest(posefile)!=cert['pose'] or digest(reference)!=cert['reference_input'] or digest(export/'mesh.node')!=cert['material_nodes']:
        raise ValueError('Force-site and material poses differ')
    pose = {r['body_index']:r['current'] for r in read(posefile)['all_bodies']}
    p, sites, wrap = [], [], 0
    for mi in range(nm):
        _, start, count, reserved = struct.unpack_from('<4I',m,muscle+164*mi)
        if reserved or count<2 or start+count>nr:
            raise ValueError('Invalid muscle range')
        for j in range(count):
            ri = start+j
            kind, si, _, _ = struct.unpack_from('<4I',m,route+16*ri)
            if kind!=1:
                if kind not in (2,3) or j in (0,count-1):
                    raise ValueError('Unexpected route geometry')
                wrap+=1
                continue
            body,*local = struct.unpack_from('<I3f',m,76+16*si)
            role='via'
            if j in (0,count-1):
                end=int(j>0)
                e,_=ep[mi,end]
                if e[2:5]!=(ri,si,body):
                    raise ValueError('Unmatched native attachment')
                local=e[8:11]
                role='insertion' if end else 'origin'
            state=pose[body]
            world=Rotation.from_quat(state['orientation_world_xyzw']).apply(local)+state['position_world_m']
            p.append(world)
            sites.append(dict(actuator=mi,name=ep[mi,0][1],route_row=ri,source_site=si,body=body,role=role))
    with np.load(reference) as z:
        face=z['face_O']
        source=z['x'][z['node_O']]
        candidate=np.loadtxt(export/'mesh.node',skiprows=1,usecols=(1,2,3),dtype='f4')[z['node_O']]
    traction, nodes=traction_nodes(t,tm,pose)
    n_sites=len(p)
    all_points=np.vstack((p,traction))
    def sample(vertices):
        surface=Enclosure(lib,vertices,face)
        try:
            return surface.query(all_points)
        finally:
            surface.close()
    old,old_d=sample(source)
    new,new_d=sample(candidate)
    patch_old,patch_new=old[n_sites:],new[n_sites:]
    patch_dist=new_d[n_sites:]
    old,old_d,new,new_d=old[:n_sites],old_d[:n_sites],new[:n_sites],new_d[:n_sites]
    labels={1:'inside',0:'outside',-1:'indeterminate'}
    counts={role:dict(Counter(labels[int(v)] for v,r in zip(new,sites) if r['role']==role)) for role in ('origin','via','insertion')}
    exterior=[]
    for i in np.flatnonzero((old!=1)|(new!=1)):
        exterior.append(dict(**sites[i],source=labels[int(old[i])],current=labels[int(new[i])],source_distance_mm=1000*old_d[i],current_distance_mm=1000*new_d[i],world_m=p[i].tolist()))
    exterior.sort(key=lambda r:(r['current']!='outside',-r['current_distance_mm'],r['route_row']))
    outside_by_mode=Counter()
    reason=Counter()
    by_index={(r['muscle_index'],r['endpoint']):(i,r) for i,r in enumerate(tm['endpoints'])}
    for e in exterior:
        if e['role']=='via':
            continue
        ei,tag=by_index[e['actuator'],e['role']]
        e['attachment']=tag['attachment_mode']
        e['admission_reason']=tag['admission_reason']
        e['distributed_node_classifications']=[labels[int(patch_new[k])] for k,node in enumerate(nodes) if node['endpoint']==ei]
        if e['current']=='outside':
            outside_by_mode[tag['attachment_mode']]+=1
            reason[tag['admission_reason']]+=1
    traction_report=dict(nodes=len(traction),source=dict(Counter(labels[int(v)] for v in patch_old)),
        candidate=dict(Counter(labels[int(v)] for v in patch_new)),changed=int(np.count_nonzero(patch_old!=patch_new)),
        outside_route_centers_by_mode=dict(outside_by_mode),outside_route_centers_by_reason=dict(reason),
        max_unit_force_error=max(r['unit_force_residual'] for r in nodes),max_unit_moment_error_m=max(r['unit_moment_residual_m'] for r in nodes),
        exceptions=[dict(**r,class_at_release=labels[int(v)],distance_mm=1000*d,world_m=q.tolist())
                    for r,v,d,q in zip(nodes,patch_new,patch_dist,traction) if v!=1])
    result=dict(traction=traction_report,articulated_pose=cert['pose'],reference_mesh=cert['reference_input'],candidate_node=cert['material_nodes'],
                muscle_payload=digest(core),tendon_payload=digest(tendon),enclosure_library=digest(lib),actuators=nm,
                named_site_occurrences=len(p),wrap_records=wrap,source=dict(Counter(labels[int(v)] for v in old)),
                candidate=dict(Counter(labels[int(v)] for v in new)),candidate_by_role=counts,
                unique_outside_world_points=len(np.unique(np.asarray(p)[new==0],axis=0)),
                changed_classifications=int(np.count_nonzero(old!=new)),exceptions=exterior,
                scope='Static named sites including route-private tendon attachments. Wrap-surface contact arcs and intersite path segments are outside this point test. Ambiguous parity is retained separately.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(nm,len(p),counts,len(exterior),flush=True)


if __name__=='__main__':
    audit(*(Path(s).resolve() for s in sys.argv[1:]))
