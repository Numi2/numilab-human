"""Compile the Float32 loaded coordinates against the unchanged material and skeleton owners."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from emit_constrained_material import write_coordinates
from material_context import sha


def bind(trial, owner, fit, out, deep_fit=None):
    job = json.loads((trial/'launch.json').read_text())
    env = job['environment']
    manifest = json.loads((owner/'identities.json').read_text())
    report = json.loads(fit.read_text())
    reference, original, t, roles, pins = inputs(trial)
    if (manifest['binary_sha256']!=sha(owner/'numi-human-native')
            or job['pins'][str(owner/'numi-human-native')]!=manifest['binary_sha256']
            or report['source']!=pins or report['proposal_sha256']!=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']):
        raise ValueError('Unmatched native registration')
    rows = np.array(report['node_ids'])
    coords = np.array(report['emitted_coordinates'],dtype='f4')
    before,after = report['installed_before'],report['installed_after']
    if (not len(rows) or not np.array_equal(np.unique(rows),rows) or not np.isfinite(coords).all()
            or after['potential_J'] is None or after['free_force_l2_N'] is None
            or not report['history']['success'] or report['history']['minimum_constraint'] < -1e-6
            or coords.shape!=(len(rows),3) or after['potential_J']>=before['potential_J']
            or after['free_force_l2_N']>=before['free_force_l2_N'] or after['bad_cells']
            or after['outside_material_bounds'] or report['reference_motion_after']['outside_material_bounds']
            or report['reference_motion_after']['bad_cells']
            or report['reference_motion_after']['maximum_J']>=report['reference_motion_before']['maximum_J']
            or report['reference_motion_after']['minimum_J']<=report['reference_motion_before']['minimum_J']
            or report['max_material_donor_moment_change_kg_m']>1e-10):
        raise ValueError('Ineligible physical registration')
    if 'loaded_mass_first_moment' in report and not report['loaded_mass_first_moment']['passed']:
        raise ValueError('Loaded mass moment would be rejected by the native material owner')
    if 'initial_volume_force_balance' in report:
        balance=report['initial_volume_force_balance']
        if not balance or np.max(abs(np.asarray(balance['emitted_dt2_acceleration'])))>.01:
            raise ValueError('Excessive emitted volume acceleration over one native increment')
    layers=report.get('node_roles', ['I']*len(rows))
    if len(layers)!=len(rows) or not set(layers)<={'I','D','O'}:
        raise ValueError('Unknown anatomical material role')
    role=dict(zip(map(int,rows),layers))
    if deep_fit is not None:
        deep=json.loads(deep_fit.read_text())
        extra=np.array(deep['node_ids'])
        if (deep['source']!=pins or deep['proposal_sha256']!=report['proposal_sha256']
                or deep['map_sha256']!=report['map_sha256'] or deep['mass_sha256']!=report['mass_sha256']
                or len(np.unique(extra))!=len(extra) or set(extra)&set(rows)
                or not deep['min_emitted_gap_on_axes_m']>0 or np.max(abs(np.array(deep['moment_change_kg_m'])))>1e-12
                or any(v['after']['bad_cells'] or v['after']['outside_material_bounds'] for v in deep['mechanical_checks'].values())):
            raise ValueError('Unmatched deep-fat registration')
        # Separate fits are composable only on disjoint element stars.
        with np.load(Path(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])) as q:
            for cells in (t[q['retained_input_cell_rows']],q['replacement_tetrahedra']):
                if (np.isin(cells,rows).any(1)&np.isin(cells,extra).any(1)).any():
                    raise ValueError('Independent registrations share an installed element')
        role.update({int(n):'D' for n in extra})
        rows=np.r_[rows,extra]
        coords=np.vstack((coords,np.asarray(deep['emitted_coordinates'],dtype='f4')))
        order=np.argsort(rows)
        rows,coords=rows[order],coords[order]
    mapping = Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    if sha(mapping)!=report['map_sha256'] or sha(mapping)!=env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_SHA256']:
        raise ValueError('Unknown shared material map')
    outer_ids=[]
    membership={i:set() for i in rows}
    with mapping.open() as stream:
        for r in csv.DictReader(stream):
            n=int(r['joint_node_id'])-1
            if r['registered_surface_layer']=='O':outer_ids.append(n)
            if n in membership:
                if r['registered_surface_layer']!=role[n] or r['source_partition'] in membership[n]:
                    raise ValueError('Not a single shared interface node')
                membership[n].add(r['source_partition'])
    if any(m!=({'fat','skin'} if role[i]=='I' else {'skin'} if role[i]=='O' else {'fat'}) for i,m in membership.items()):
        raise ValueError('Missing material owner')
    if 'O' in role.values():
        bed=report['registered_bed']
        outer=sorted(i for i in rows if role[i]=='O')
        loaded=[i for i in outer if original[i,2]<bed['width_m']]
        if outer!=bed['outer_nodes'] or loaded!=bed['loaded_nodes'] or bed['native_added_constraints']!=0:
            raise ValueError('Incomplete registered bed material')
        preserve_z=sorted(set(outer)-set(loaded)) if bed.get('redistributed') else outer
        if bed.get('redistributed') and (abs(bed['reaction_drift_N'])>.002 or np.max(np.abs(bed['moment_drift_Nm']))>.001):
            raise ValueError('Change to initial support wrench')
        if not np.array_equal(original[preserve_z,2],coords[np.searchsorted(rows,preserve_z),2]) or not np.array_equal(original[loaded,:2],coords[np.searchsorted(rows,loaded),:2]):
            raise ValueError('Initial contact application points or gaps moved')
    inc='NumiHumanLoadedSkinPoseData.inc'
    header=(owner/inc).read_text()
    anchor_table=re.search(r'constexpr Anchor anchors\[\]=\{(.*?)\};',header,re.S)
    anchors=[int(a) for a in re.findall(r'\{\s*\d+u?\s*,\s*(\d+)u?\s*,',anchor_table.group(1))]
    if len(anchors)!=14 or set(anchors).intersection(rows):
        raise ValueError('Joint attachment changed')
    key='NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'
    from supported_material_admission import check_installed
    original[rows] = coords
    admission = check_installed(reference, original, t, roles, env, owner)
    manifest['installed_tissue_admission'] = admission
    out.mkdir(exist_ok=False)
    new=out/'current.node'
    write_coordinates(Path(env[key]),new,rows,coords)
    if not np.array_equal(np.loadtxt(new,skiprows=1,usecols=(1,2,3)),original):
        raise ValueError('Unexpected emitted coordinates')
    bed = report.get('registered_bed', {})
    compile_loaded_owner(owner, out, fit, manifest, old=pins['installed_pose_sha256'], nodes=rows,
                         footprint=(outer_ids, original, bed['first_contact_m'], bed['width_m']) if bed.get('redistributed') else None,
                         deep_fit=deep_fit)


def compile_loaded_owner(owner, out, fit, manifest, old, nodes, footprint=None, deep_fit=None):
    new = out/'current.node'
    inc = 'NumiHumanLoadedSkinPoseData.inc'
    key = 'NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'
    header = (owner/inc).read_text()
    case = out/'native-owner'
    if case.exists() or sha(owner/'numi-human-native') != manifest['binary_sha256']:
        raise ValueError('Existing output or unmatched source binary')
    subprocess.run(['cp','-cR',str(owner),str(case)],check=True)
    if header.count(old)!=1:
        raise ValueError('Unknown native pose pin')
    pin=sha(new)
    (case/inc).write_text(header.replace(old,pin))
    if footprint is not None:
        from material_bed_release import bind_contact_footprint
        outer, current, first, width = footprint
        occupied = bind_contact_footprint(case, outer, current, first, width)
        manifest['registered_bed_footprint'] = dict(contact_nodes=len(occupied), minimum_gap_m=first,
            outer_nodes=len(outer), loader_sha256=sha(case/'NumiHumanLoadedSkinPose.hpp'))
    cmds=[[a.replace(str(owner)+'/',str(case)+'/') for a in cmd] for cmd in manifest['commands']]
    for cmd in cmds:
        subprocess.run(cmd,check=True)
    binary=sha(case/'numi-human-native')
    previous=manifest.get('loaded_pose_override')
    if previous and (previous['sha256']!=old or previous['key']!=key or sha(Path(previous['path']))!=old):
        raise ValueError('Previous loaded coordinate owner changed')
    parent=previous['parent_sha256'] if previous else old
    manifest.update(commands=cmds,binary=binary,binary_sha256=binary,binary_path=str(case/'numi-human-native'),
                    parent_pose_owner_sha256=sha(owner/'numi-human-native'),
                    loaded_pose_override=dict(key=key,parent_sha256=parent,immediate_parent_sha256=old,path=str(new),sha256=pin,
                                              header_sha256=sha(case/inc),fit_sha256=sha(fit),nodes=nodes.tolist()))
    if deep_fit is not None:
        manifest['loaded_pose_override']['deep_fit_sha256']=sha(deep_fit)
    (case/'identities.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    bind(*(Path(arg).resolve() for arg in sys.argv[1:]))
