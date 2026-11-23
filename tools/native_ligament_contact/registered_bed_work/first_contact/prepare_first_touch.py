"""Register the whole adult at first bed contact before applying its physical loads."""
import csv
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from emit_constrained_material import write_coordinates
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import check_installed


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def block(text, kind, name):
    return re.search(r'constexpr ' + kind + ' ' + name + r'\[\]=\{\n(.*?)\n\};', text, re.S)[1]


def bodies(text):
    positions = {}
    rotations = {}
    for line in text.splitlines():
        body = int(re.match(r'\{(\d+),', line)[1])
        position, rotation = re.findall(r'\{([^{}]+)\}', line)
        for raw, out in ((position, positions), (rotation, rotations)):
            out[body] = np.array([float.fromhex(s.removesuffix('f')) for s in raw.split(',')])
    return positions, rotations


def vector(v):
    return '{' + ','.join(float(np.float32(e)).hex()+'f' for e in v) + '}'


def prepare(parent, out, audit, predicate, material_origin=False):
    out.mkdir(exist_ok=False)
    cfg = read(parent / 'launch.json')
    env = cfg['environment']
    old = Path(cfg['argv'][0]).parent
    ids = read(old / 'identities.json')
    header = 'NumiHumanLoadedSkinPoseData.inc'
    loader = ids['loaded_pose_override']
    if sha(old / header) != loader['header_sha256'] or sha(Path(cfg['argv'][0])) != ids['binary']:
        raise ValueError('Parent binary or compiled registration changed')
    x, prior_pose, cells, roles, pins = inputs(parent)
    registered = Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE'])
    if sha(registered) != pins['reference_sha256'] or loader['sha256'] != pins['installed_pose_sha256']:
        raise ValueError('Reference or prior loaded pose changed')
    text = (old / header).read_text()
    reference, _ = bodies(block(text,'Body','reference'))
    target_block = block(text,'Body','target')
    target, q = bodies(target_block)
    _, refq = bodies(block(text,'Body','reference'))
    shifts = np.array([reference[i] - target[i] for i in target])
    root_idx = cfg['argv'].index('--root-pose')
    root_z = float(cfg['argv'][root_idx+3])
    dz = float(np.float32(reference[0][2])) - root_z
    if np.max(abs(shifts - [0,0,dz])) > 2e-7 or any(np.max(abs(q[i]-refq[i])) > 2e-7 for i in target):
        raise ValueError('Donor poses do not differ by one vertical translation')
    map_prefix = 'NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_'
    mp = Path(env[map_prefix+'CSV'])
    if sha(mp) != env[map_prefix+'SHA256']:
        raise ValueError('Unregistered outer boundary')
    with mp.open() as f:
        outer = np.array([int(r['joint_node_id'])-1 for r in csv.DictReader(f) if r['registered_surface_layer']=='O'])
    old_contact = re.search(r'registeredBedContactNodes\[\]=\{([^}]+)\}',text)[1]
    width = float.fromhex('0x1.4f8b580000000p-17')
    old_ids = np.asarray([int(s.rstrip('u')) for s in old_contact.split(',')])
    if not np.array_equal(np.sort(outer[prior_pose[outer,2] < width]),old_ids):
        raise ValueError('Parent contact footprint differs')
    # An exactly representable offset keeps the registered sliver edges intact
    # across Float32 binades and keeps the closest skin inside the bed barrier.
    clearance_offset = 2.0**-19
    bed_z = -clearance_offset if material_origin else 0.
    shift = clearance_offset + bed_z
    cur = x.copy()
    cur[:,2] = (x[:,2].astype(float) + shift).astype('f4')
    if not 0 < float(cur[:,2].min())-bed_z < width or cur[:,2].min()!=cur[outer,2].min():
        raise ValueError('Native bed separation changed')
    contact = np.sort(outer[cur[outer,2].astype(float)-bed_z < width])
    if not len(contact):
        raise ValueError('No physical bed footprint')
    target_file = out/'current.node'
    write_coordinates(registered,target_file,np.arange(len(cur)),cur)
    emitted = np.loadtxt(target_file,skiprows=1,usecols=(1,2,3))
    if not np.array_equal(emitted,cur):
        raise ValueError('Native position precision differs')
    material = check_installed(x,cur,cells,roles,env,old)
    if sha(predicate)!=read(parent/'accepted-layers.json')['library_sha256']:
        raise ValueError('Mismatched intersection predicate')
    construction=out/'construction.json'
    save(construction,dict(cfg,pins=cfg['pins']|{str(target_file):sha(target_file)}))
    audit_env=dict(os.environ,PYTHONPATH=str(predicate.parent)+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable,str(audit),str(construction),str(predicate),str(out/'layers.json'),
                    '--constructed-nodes',str(registered),str(target_file)],check=True,env=audit_env)
    if np.any(read(out/'layers.json')['pair_counts']):
        raise ValueError('Outer, bonded or deep layer intersections')
    pose = '\n'.join('{' + str(i) + ',' + vector(reference[i]+[0,0,shift]) + ',' + vector(q[i]) + '},' for i in target)
    text = text.replace(target_block,pose)
    anchor_block = block(text,'Anchor','anchors')
    new_anchors = []
    for line in anchor_block.splitlines():
        stable, node, donor = map(int,re.findall(r'(\d+)u',line))
        prev = np.array([float.fromhex(e.removesuffix('f')) for e in re.findall(r'\{([^{}]+)\}',line)[0].split(',')])
        if np.linalg.norm(prev+[0,0,dz]-x[node])>2e-7:
            raise ValueError(f'Anchor {stable} changes its position on body {donor}')
        new_anchors.append('{%du,%du,%du,%s},' % (stable,node,donor,vector(cur[node])))
    text=text.replace(anchor_block,'\n'.join(new_anchors)).replace(old_contact,','.join(str(i)+'u' for i in contact))
    if text.count(loader['sha256'])!=1:
        raise ValueError('Ambiguous pose hash')
    text=text.replace(loader['sha256'],sha(target_file))
    dest=out/'owner'
    subprocess.run(['cp','-cR',str(old),str(dest)],check=True)
    gap = (float(cur[outer, 2].min())-bed_z).hex()
    (dest/header).write_text(text + '\nnamespace numiLoadedData { constexpr double minimumBedGap = '+gap+'; constexpr double fixedBedHeight = '+bed_z.hex()+'; }\n')
    main = 'numilab_human_myosim_visual_probe.mm'
    for filename, count in [('NumiHumanLoadedSkinPose.hpp', 2), (main, 1)]:
        source = (dest/filename).read_text()
        literal = '0x1.afd3ae0000000p-18'
        if source.count(literal) != count:
            raise ValueError('Unexpected native bed certificate')
        (dest/filename).write_text(source.replace(literal, 'numiLoadedData::minimumBedGap'))
    support_path=Path(cfg['argv'][cfg['argv'].index('--support-contact-payload')+1])
    support_sha=sha(support_path)
    if material_origin:
        # Change only the reference frame of the existing, fixed NHCNT plane.
        # Contact eligibility, friction, normals and the 32 legacy witnesses are unchanged.
        raw=bytearray(support_path.read_bytes())
        if struct.unpack_from('<8s4I',raw)!=(b'NHCNT1\0\0',1,len(reference),32,0) or struct.unpack_from('<7f',raw,56)!=(0.,0.,0.,0.,0.,1.,1.):
            raise ValueError('Unsupported source bed for a rigid change of world origin')
        struct.pack_into('<f',raw,64,bed_z)
        support_path=out/'reference-frame-bed.nhcnt'
        support_path.write_bytes(raw)
        loader_path=dest/'NumiHumanLoadedSkinPose.hpp'
        body=loader_path.read_text()
        for before,after in [
            ('p[2]>=0.', 'p[2]>=numiLoadedData::fixedBedHeight'),
            ('bed.groundPointZ!=0', 'bed.groundPointZ!=numiLoadedData::fixedBedHeight'),
            ('std::min(minimum,v[2])','std::min(minimum,v[2]-bed.groundPointZ)'),
            ('const double gap=p.at(node)[2];','const double gap=p.at(node)[2]-numiLoadedData::fixedBedHeight;'),
            (' fixed_bed_offset_m=0 minimum_skin_gap_m=', ' fixed_bed_offset_m="<<numiLoadedData::fixedBedHeight<<" minimum_skin_gap_m='),
        ]:
            if body.count(before)!=1:raise ValueError('Unknown signed bed certificate: '+before)
            body=body.replace(before,after)
        loader_path.write_text(body)
    commands=[[v.replace(str(old)+'/',str(dest)+'/') for v in cmd] for cmd in ids['commands']]
    if str(dest/'numilab_human_myosim_visual_probe.mm') not in commands[0] or str(dest/'numi-human-native') not in commands[1]:
        raise ValueError('Rebuild would address a different owner')
    for cmd in commands:
        subprocess.run(cmd,check=True,cwd=cfg['cwd'])
    exe=dest/'numi-human-native'
    save(dest/'identities.json',dict(ids,binary=sha(exe),binary_sha256=sha(exe),binary_path=str(exe),commands=commands,source=sha(dest/main),
        generated_header=sha(dest/header),installed_tissue_admission=material,parent_pose_owner_sha256=sha(Path(cfg['argv'][0])),
        loaded_pose_override=dict(path=str(target_file),sha256=sha(target_file),header_sha256=sha(dest/header),
        immediate_parent_sha256=pins['installed_pose_sha256'],reference_sha256=pins['reference_sha256'])))
    save(out/'installation.json',dict(mode='first contact under normal gravity',parent_launch=sha(parent/'launch.json'),
        source=pins,material=material,loaded_pose=sha(target_file),parent=loader['sha256'],binary=sha(exe),
        root_z_m=float(np.float32(root_z+dz+shift)),skeletal_rise_m=dz+shift,material_translation_m=shift,
        initial_gap_m=float(cur[:,2].min())-bed_z,original_support_sha256=support_sha,support=str(support_path),support_sha256=sha(support_path),bed_z_m=bed_z,previous_contact_vertices=len(old_ids),contact_vertices=contact.tolist(),
        previous_distortion_vertices_above_1mm=int(np.count_nonzero(np.linalg.norm(prior_pose.astype(float)-x+[0,0,dz],axis=1)>.001)),
        anchors=len(new_anchors),same_source_mass_ledger=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256'],
        layer_pairs=read(out/'layers.json')['pair_counts']))


if __name__=='__main__':
    args=sys.argv[1:]
    exact=args[-1]=='--material-origin'
    prepare(*(Path(v).resolve() for v in (args[:-1] if exact else args)),material_origin=exact)
