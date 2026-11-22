"""Match a one-root supported contact transaction to its anatomically registered dense O boundary."""
import argparse
import csv
import json
from pathlib import Path
import hashlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from registered_bed_traction import footprint, visible_samples, loads_on_surface, pressure_minimax


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def evaluate(run, full, ray_tool, output):
    native = read(run / 'launch.json')
    registered = read(full / 'launch.json')
    terms = read(run / 'terminal.receipt.json')
    proof = read(run / 'execution.json')
    assert proof['native_exit_code'] == 0 and terms['accepted_step'] == 1
    env = registered['environment']
    files = {}
    rigid = []
    beds = []
    for job in (native, registered):
        rk = job['argv'][1]
        bed = job['argv'][job['argv'].index('--support-contact-payload') + 1]
        assert job['pins'][rk] == digest(Path(rk))
        bed_path = Path(bed).resolve(strict=True)
        manifest = read(bed_path.parent/'resting-supine-scene.manifest.json')['outputs']['support_contact']
        bed_hash = digest(bed_path)
        assert Path(manifest['path']).resolve()==bed_path and manifest['sha256']==bed_hash
        assert manifest['record_count']==32 and manifest['bytes']==bed_path.stat().st_size
        assert all(value==bed_hash for key,value in job['pins'].items() if Path(key).name==bed_path.name)
        rigid.append(job['pins'][rk])
        beds.append(bed_hash)
    assert len(set(rigid)) == len(set(beds)) == 1
    def pinned(field):
        base = 'NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_' + field
        f = Path(env[base + '_CSV'])
        files[field] = digest(f)
        if files[field] != env[base + '_SHA256']:
            raise ValueError('Registration identity differs: ' + field)
        return f
    surface = np.loadtxt(pinned('SURFACE_O_NODES'), delimiter=',', skiprows=1,
                         usecols=(0,1,2,3,4,8,9,10))
    assert np.array_equal(surface[:,0], np.arange(len(surface)))
    rest = surface[:, 2:5]
    owner = surface[:, 1].astype(int)
    local = surface[:, 5:8]
    faces = np.loadtxt(pinned('SURFACE_O_FACES'), delimiter=',', skiprows=1,
                       usecols=(1,2,3), dtype=int)
    if faces.min() < 0 or faces.max() >= len(rest):
        raise ValueError('invalid outer surface connectivity')
    nodes = np.full(len(rest), -1)
    with pinned('COMBINED_NODE_MAP').open() as inp:
        for row in csv.DictReader(inp):
            if row['registered_surface_layer'] == 'O':
                i = int(row['registered_surface_vertex_row'])
                assert nodes[i] == -1 and int(row['donor_body']) == owner[i]
                nodes[i] = int(row['joint_node_id']) - 1
    assert np.min(nodes)>=0 and len(np.unique(nodes))==len(nodes)
    frames = {int(r['donor']):r for r in table(pinned('DONOR_FRAMES'))}
    posed = np.empty_like(rest)
    initial = {r['body_index']:r for r in terms['initial_anatomical_registration']['body_poses']}
    max_error = 0.
    for k in np.unique(owner):
        rows = owner == k
        ref = frames[k]
        mat = np.array([float(ref[f'current_R{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
        origin = np.array([float(ref['current_' + a]) for a in 'xyz'])
        residual = rest[rows] - (local[rows] @ mat.T + origin)
        max_error = max(max_error, np.linalg.norm(residual, axis=1).max())
        target = initial[k]
        rot = Rotation.from_quat(target['quaternion_xyzw']).as_matrix()
        posed[rows] = (local[rows] + np.linalg.solve(mat, residual.T).T) @ rot.T + target['position_m']
    if max_error > 1e-6:
        raise ValueError(('inconsistent reference frames',max_error))
    # All forces and query coordinates belong to the same single physical root.
    impulses = sorted(table(run / 'resting-com-support-impulses.csv'), key=lambda r:int(r['contact_index']))
    selected = {}
    for point in table(run / 'resting-support-selected-skin-witness.csv'):
        key = int(point['region'])
        if key in selected and selected[key]!=point:
            raise ValueError('ambiguous selection for a single physical root')
        selected[key]=point
    points=[selected[k] for k in sorted(selected)]
    if len(impulses) != len(points) or not points:
        raise ValueError('select every step in the support geometry callback to pair the actual load application point')
    dt = float(terms['accepted_time_s'])
    actual = []
    sections = []
    for i,(row,point) in enumerate(zip(impulses,points)):
        assert int(row['sample_start_step'])==int(point['pre_dynamics_step_index'])==0
        assert int(row['segment_steps'])==int(row['accepted_step'])==int(point['accepted_step'])==1
        assert int(point['region'])==int(row['contact_index'])==i
        assert point['source_geometry_index']==row['source_geometry_index']
        assert all(int(point[k])==0 for k in ('region_error','global_error','nonfinite_jacobian_count'))
        assert row['row_class']=='external_support_contact'
        assert np.array_equal([float(row['normal_'+a]) for a in 'xyz'], [0.,0.,1.])
        actual.append(float(row['normal_impulse_ns']) / dt)
        sections.append([float(point['selected_point_world_'+a+'_m']) for a in 'xyz'])
    actual, sections = np.asarray(actual), np.asarray(sections)
    assert np.isfinite(actual).all() and np.min(actual) >= 0 and actual.sum()>0
    from scipy.spatial import cKDTree
    distance, closest = cKDTree(posed).query(sections)
    selected_vertex = np.array([int(q['selected_skin_vertex_local']) for q in points])
    on_material = native['environment'].get('NUMI_HUMAN_LOCAL_BED_REGISTERED_OUTER')=='1'
    material_error = None
    if on_material:
        material_error = float(np.max(np.linalg.norm(posed[selected_vertex]-sections,axis=1)))
        if material_error>1.e-6:raise ValueError(('selector disagrees with registered O node',material_error))
    weighted = actual @ sections[:,:2] / actual.sum()
    # Use the existing precision-tested full-surface intersection routine.
    sys.path.insert(0, str(ray_tool.parent))
    from prepare_bed_columns import hits
    from collections import defaultdict
    states={}
    output.mkdir(parents=True, exist_ok=True)
    for name,x in [('release',rest), ('supported_rigid_prediction',posed)]:
        sample = visible_samples(footprint(x,faces,nodes), x,faces,hits)
        _, areas, bary, locations, ids = sample
        rows, mapped = loads_on_surface(sample, sections, actual)
        if mapped:
            node_load = defaultdict(float)
            for ind,weights,loads in mapped:
                for n,val in zip(ind.ravel(), (weights * loads[:,None]).ravel()):
                    node_load[int(n)] += float(val)
            keys = np.array(sorted(node_load))
            vals = np.array([node_load[int(i)] for i in keys])
            pos = x[np.argsort(nodes)[np.searchsorted(nodes[np.argsort(nodes)],keys)]]
            feasible = np.array([r['section'] for r in rows if r['moment_feasible']],dtype=int)
            error_N = float(abs(vals.sum()-actual[feasible].sum()))
            error_Nm = float(np.max(abs(vals @ pos[:,:2] - actual[feasible] @ sections[feasible,:2])))
            assert error_N <= 1e-6 and error_Nm <= 1e-6
            with (output/(name+'-normal-guess.csv')).open('w') as f:
                writer=csv.writer(f);writer.writerow(['skin_fat_node_zero_based','initial_trial_normal_N'])
                writer.writerows(zip(keys,vals))
        else:
            error_N = error_Nm = None
        global_fit=pressure_minimax(locations[:,:2],areas,weighted,actual.sum())
        states[name]=dict(exposed_area_m2=float(areas.sum()),sample_count=len(areas),minimum_uncompressed_gap_m=float(x[:,2].min()),
                          immersed_vertices=int((x[:,2]<0).sum()),source_rows=rows,
                          recoverable_section_load_N=float(sum(r['normal_N'] for r in rows if r['moment_feasible'])),
                          independent_wrench_feasible=global_fit is not None,
                          minimum_global_peak_pressure_Pa=None if global_fit is None else global_fit[1],
                          mapped_force_error_N=error_N, mapped_moment_error_Nm=error_Nm)
    report=dict(dense_release_hash=digest(full/'launch.json'),native_launch_hash=digest(run/'launch.json'),
                source_evidence={n:digest(run/n) for n in ('terminal.receipt.json','resting-com-support-impulses.csv','resting-support-selected-skin-witness.csv','execution.json')},
                source_geometry=files,skeleton_sha256=rigid[0],bed_sha256=beds[0],max_source_frame_error_m=float(max_error),
                selected_outer_vertex_reconstruction_error_m=material_error,nearest_outer_node_distance_m=distance.tolist(),dt_s=dt,total_normal_N=float(actual.sum()),active_rows=int((actual>0).sum()),
                center_of_normal_impulse_xy_m=weighted.tolist(),evaluations=states,
                scope='One native section impulse per row. Rigid donor poses are an uncompressed trial for the dense volume. Face quadrature is clipped at the bed; node loads are construction guesses only. Each separate region has nearest-witness ownership in the bed plane. No volume displacement or force has been applied.')
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print([(name,s['exposed_area_m2'],s['recoverable_section_load_N'],s['minimum_global_peak_pressure_Pa']) for name,s in states.items()],report['total_normal_N'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','full','ray_tool','output'):p.add_argument(name,type=Path)
    arg=p.parse_args();evaluate(arg.run,arg.full,arg.ray_tool,arg.output)
