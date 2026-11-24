"""Compare the native row attribution against the authority that rejected this step."""
import argparse
import csv
import json
import math
import re
from pathlib import Path
from compile_connected_joint import digest

def read_json(path):
    return json.loads(path.read_bytes())

def read_terminal(run):
    certificates = []
    work = []
    rows = []
    for line in (run/'stderr.log').read_text().splitlines():
        if line.startswith('certificate_row_measure '):
            v = dict(kv.split('=', 1) for kv in line.split()[1:])
            rows.append({k:(float(val) if k.endswith('2') else int(val)) for k,val in v.items()})
        for prefix, destination in [('fem_newton_terminal_certificate=',certificates),('matter_fgmres_active_work=',work)]:
            if line.startswith(prefix):
                v = json.loads(line[len(prefix):])
                if v['root'] == 1:
                    destination.append(v)
    if len(certificates)!=1 or len(work)!=1:
        raise ValueError('Missing or ambiguous terminal state')
    f=read_json(run/'failure.json')
    with (run/'native-traces/resting-coupled.csv').open() as stream:
        trace=list(csv.DictReader(stream))
    return certificates[0],work[0],rows,dict(accepted_s=[float(r['time_s']) for r in trace],segment_start=f['segment_start'],status=f['matter_code'],reason=f['matter_diagnostics'][3])

def run(control, candidate, mapping, out):
    a, aw, _, ca=read_terminal(control)
    b, bw, spans, cb=read_terminal(candidate)
    if ca!=cb or ca['status']!=10 or ca['reason']!=-902 or len(ca['accepted_s'])!=1:
        raise ValueError('Acceptance or rejection differed')
    if len({r['span'] for r in spans})!=len(spans) or not spans or any(r['root']!=1 or r['local']>=r['count'] for r in spans):
        raise ValueError('Invalid span receipt')
    raw_norm=math.sqrt(math.fsum(r['mechanics2']+r['pressure2'] for r in spans))
    norm=raw_norm/max(1.0,b['fgmres'][0]['diagnostics'][1])
    official=b['objects'][0]['nonlinear'][0]
    if not math.isclose(norm,official,rel_tol=2e-5) or any(o['nonlinear'][0]!=official for o in b['objects']):
        raise ValueError(f'Incomplete attribution: {norm:g} vs {official:g}')
    reference=a['objects'][0]['nonlinear'][0]
    if not math.isclose(reference,official,rel_tol=1e-5):
        raise ValueError('Observer or replay changed terminal residual')
    log=(candidate/'stdout.log').read_text()
    skin=re.search(r'whole_body_skin_fat_self_contact=.*? object=(\d+) nodes=(\d+)',log)
    if not skin:raise ValueError('Missing registered skin/fat owner')
    labels={}
    for match in re.finditer(r'resting_fascia_object=compiled_reference_mode object=\d+ state_offset=(\d+) state_count=(\d+).*? region_begin=(\d+) region_end=(\d+)',log):
        labels[tuple(map(int,match.group(1,2)))]=f'fascia regions {match[3]} through {int(match[4])-1}'
    skin_id=int(skin[1])
    for r in spans:
        r['fraction_squared_norm']=(r['mechanics2']+r['pressure2'])/raw_norm**2
        r['peak_fraction_squared_norm']=r['peak2']/raw_norm**2
        r['label']=labels.get((r['offset'],r['count']),'FEM object '+str(r['span']))
        if r['span']==skin_id:
            if r['count']!=int(skin[2]):raise ValueError('Skin/fat row count mismatch')
            r['label']='registered bonded skin and fat'
            with mapping.open() as f:
                row=next(row for row in csv.DictReader(f) if int(row['joint_node_id'])==r['local']+1)
            r['peak_source']={k:row[k] for k in ('joint_node_id','source_partition','role','donor_body','anatomical_region_id','incident_material_mask')}
        elif r['span']==2**32-2:
            r['label']='articulated generalized velocity'
            with (candidate/'native-traces/resting-com-q-index-map.csv').open() as f:
                r['peak_dof']=next((d for d in csv.DictReader(f) if d['local_v_index']==str(r['local'])),None)
        elif r['span']==2**32-3:
            r['label']='support contact'
    keys=('used_columns_sum','active_newton_begins','budget_exhausted_solves','last_fgmres_residual_norm','minimum_newton_norm','last_newton_correction_ratio')
    data=dict(control_sha256=digest(control/'launch.json'),observed_sha256=digest(candidate/'launch.json'),
      control_outcome=ca,observed_outcome=cb,control_residual=reference,native_residual=official,
      reconstructed_residual=norm,relative_error=abs(norm-official)/official,
      control_solver={k:aw[k] for k in keys},observed_solver={k:bw[k] for k in keys},
      xyz_rigid_support_fraction=math.fsum(r['mechanics2'] for r in spans)/raw_norm**2,
      pressure_fraction=math.fsum(r['pressure2'] for r in spans)/raw_norm**2,
      spans=sorted(spans,key=lambda r:-r['fraction_squared_norm']))
    out.write_text(json.dumps(data,indent=2)+'\n')
    print('reconstructed',norm,'official',official,'top',data['spans'][0],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('control','candidate','mapping','out'):p.add_argument(n,type=Path)
    run(**vars(p.parse_args()))
