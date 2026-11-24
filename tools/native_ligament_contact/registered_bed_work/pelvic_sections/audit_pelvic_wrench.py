"""Compare emitted native section maps against independent unilateral material columns."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

def read_table(folder, name):
    with (folder / ('resting-local-bed-' + name + '.csv')).open() as stream:
        return list(csv.DictReader(stream))

def sections(folder):
    return {int(r['contact_index']): r for r in read_table(folder, 'sections')}

def footprint(folder):
    data = read_table(folder, 'footprint')
    result = {int(r['vertex']): r for r in data}
    if len(result) != len(data):
        raise ValueError('Repeated contact column')
    return result

def section_stiffness(row):
    return 1 / (float(row['skin_compliance_m_per_n']) + float(row['fat_compliance_m_per_n']))

def audit(control, candidate):
    old, new = footprint(control), footprint(candidate)
    s0, s1 = sections(control), sections(candidate)
    if old.keys() != new.keys() or set(s0) != set(range(38)) or set(s1) != set(range(44)):
        raise ValueError('Unmatched native footprint or unexpected support rows')
    families = {6: {6,38,39,40}, 7: {7,41,42,43}}
    for v, row in old.items():
        after = new[v]
        if any(row[k] != after[k] for k in row if k != 'row'):
            raise ValueError(f'Changed physical column {v}')
        parent = int(row['row'])
        if int(after['row']) not in families.get(parent, {parent}):
            raise ValueError(f'Source material reassigned across footprints at {v}')
    for row in set(s0) - families.keys():
        if s0[row] != s1[row]:
            raise ValueError(f'Unrelated contact row {row} changed')
    records, sources = [], {}
    for parent, children in families.items():
        ids = sorted(i for i in old if int(old[i]['row']) == parent)
        cols = [old[i] for i in ids]
        xyz = np.array([[float(r[k]) for k in ('x_m','y_m','z_m')] for r in cols])
        area = np.array([float(r['area_m2']) for r in cols])
        stiffness = area / np.array([float(r['skin_m'])/float(s0[parent]['skin_modulus_pa']) +
                                     float(r['fat_m'])/float(s0[parent]['fat_modulus_pa']) for r in cols])
        group = np.array([int(new[i]['row']) for i in ids])
        sources[parent] = dict(column_count=len(cols), total_area_m2=float(area.sum()),
                               stiffness_N_per_m=float(stiffness.sum()), rows=sorted(children))
        if set(group) != children:
            raise ValueError('Missing native child footprint')
        for row_set, table in [({parent},s0),(children,s1)]:
            for key, field, arr in [('area',lambda r:float(r['projected_area_m2']),area),
                                    ('stiffness',section_stiffness,stiffness)]:
                for r in row_set:
                    mask = np.ones(len(cols),bool) if table is s0 else group == r
                    if not math.isclose(field(table[r]), float(arr[mask].sum()), rel_tol=1e-6):
                        raise ValueError(f'{key} mismatch at row {r}')
        offset = xyz - np.average(xyz,axis=0,weights=area)
        for angle in (0.,-.02,.02):
            n = np.array([math.sin(angle),0.,math.cos(angle)])
            h = offset @ n
            arm = np.cross(offset,n)
            for indent in (.002,.004,.006):
                level = h.min() + indent
                f = stiffness * np.maximum(level-h,0.)
                force, moment = f.sum(), f @ arm
                res = dict(parent=parent, tilt_rad=angle, indent_m=indent,
                           independent_force_N=float(force), independent_moment_Nm=moment.tolist())
                for key, labels in [('control',np.full(len(ids),parent)), ('candidate',group)]:
                    F = 0.
                    M = np.zeros(3)
                    for r in sorted(set(labels)):
                        ix = np.flatnonzero(labels == r)
                        j = ix[np.argmin(h[ix])]
                        normal_force = stiffness[ix].sum()*max(level-h[j],0.)
                        F += normal_force
                        M += arm[j]*normal_force
                    res[key+'_force_error_N'] = float(abs(F-force))
                    res[key+'_moment_error_Nm'] = float(np.linalg.norm(M-moment))
                records.append(res)
    fails = [{k:r[k] for k in ('parent','tilt_rad','indent_m')} | {'metric':k}
             for r in records for k in ['force_error_N','moment_error_Nm']
             if r['candidate_'+k] > r['control_'+k]*(1+1e-6)+1e-8]
    return dict(reference='Positive-area emitted columns, unilateral static normal response; reference is internal to the boundary material law.', scope='No tangential traction or dynamic clearance in this comparison.', sources=sources,
                all_source_columns=len(old), cases=records, failures=fails, passed=not fails)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('control','candidate','output'): p.add_argument(a,type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(audit(a.control,a.candidate),indent=2)+'\n')
