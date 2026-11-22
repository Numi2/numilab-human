"""Cross-check the native line search against all installed material cells on matched accepted motion."""
import json
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha

def run(root, accepted=None, rejected=None, screen_both=False, forecast_steps=1):
    accepted=accepted or root/'paired-fat-three-resumed'
    rejected=rejected or root/'four-physical-resumed'
    x,start,t,tags,pins=inputs(accepted)
    a=json.loads((accepted/'launch.json').read_text())
    b=json.loads((rejected/'launch.json').read_text())
    if a['pins']!=b['pins']:
        raise ValueError('Material or executable mismatch across launches')
    env=a['environment']
    mesh=Path(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    flow=root/'paired-fat-material/fine-mass-routed/current-density-closed-flow.npz'
    mapping=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV'])
    old=json.loads((root/'current-cell-witnesses.json').read_text())
    if old['source']['elements_sha256']!=pins['elements_sha256']:
        raise ValueError('Cannot reuse element offset')
    offset=old['source_element_offset']
    endpath=accepted/'physical-accepted.node'
    endpoint=json.loads((accepted/'verified-endpoint.json').read_text())
    admission=json.loads((accepted/'admission.json').read_text())
    receipt=json.loads((accepted/'terminal.receipt.json').read_text())
    if (receipt['physical_endpoint']!='accepted' or endpoint['frame_sha256']!=receipt['pack_file_sha256']
            or endpoint['physical_nodes_sha256']!=sha(endpath) or admission['accepted_steps']!=receipt['accepted_step']):
        raise ValueError('Unmatched accepted material snapshot')
    end=np.loadtxt(endpath,skiprows=1,usecols=(1,2,3))
    fail=json.loads((rejected/'failure.json').read_text())
    root_index=fail['segment_start']
    if root_index<=0 or root_index!=admission['accepted_steps'] or fail['matter_code']!=10:
        raise ValueError('Different physical failure')
    trace=rejected/'stderr.log'
    logs=[json.loads(l.split('=',1)[1]) for l in trace.read_text().splitlines() if l.startswith('fem_backtracking_trials=')]
    log=next(l for l in reversed(logs) if l['root']==root_index and l['object']==15)
    full=next(v for v in log['trials'] if v['alpha']==1)
    if not full['tile_witnesses_complete']:
        raise ValueError('Incomplete native witnesses')
    native=sorted({int(i)-offset for i in full['tile_witness_tets']})
    with np.load(mesh) as p:
        keep=p['retained_input_cell_rows']
        t=np.vstack((t[keep],p['replacement_tetrahedra']))
        po=p['replacement_parent_offsets']
        parents=p['replacement_parent_cell_rows']
    if forecast_steps<1 or (not screen_both and forecast_steps!=1):raise ValueError('Invalid geometric lookahead')
    frame=[x.astype(float),start.astype(float),end,end+(end-start)*(forecast_steps/root_index)]
    rows=[]; hits=[]
    for u in range(0,len(t),65536):
        cells=t[u:u+65536]
        def determinants(q):
            e=q[cells[:,1:]]-q[cells[:,:1]]
            return np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2]))
        ref=determinants(frame[0])
        if np.any(ref<=0):raise ValueError('Inverted reference')
        states=np.column_stack([determinants(q)/ref for q in frame[1:]])
        hit=np.flatnonzero((states[:,-1]<.4)|(states[:,-1]>2.4)) if screen_both else np.flatnonzero(states[:,-1]<.36)
        hits.extend((hit+u).tolist())
        for i in sorted(set(native)|set((hit+u).tolist())):
            if u<=i<u+len(cells):
                source=[int(keep[i])] if i<len(keep) else parents[po[i-len(keep)]:po[i-len(keep)+1]].astype(int).tolist()
                rows.append(dict(installed_row=i,global_element=i+offset,node_ids=t[i].tolist(),source_rows=source,J=states[i-u].tolist()))
    if not screen_both and hits!=native:
        raise ValueError(f'Model prediction disagrees with GPU witnesses {hits} {native}')
    result=dict(geometry=pins,mesh_sha256=sha(mesh),mass_sha256=sha(flow),map_sha256=sha(mapping),
        physical_endpoint_sha256=sha(endpath),native_trace_sha256=sha(trace),installed_witness_rows=sorted(set(native)|set(hits)),
        accepted_steps=root_index,material_rows=len(t),witnesses=rows,selected_alpha=log['selected_alpha'],
        phases=['installed','three accepted steps','fourth-step secant diagnostic'],native_full_trial=full,
        secant_scope='Diagnostic linear extension of accepted displacements. Only native GPU steps advance physical time.')

    if screen_both:
        result.update(native_witness_rows=native, predicted_rows=hits, forecast_limits=[.4,2.4], forecast_steps=forecast_steps)
        result['phases']=['installed',f'{root_index} accepted steps',f'{forecast_steps} additional steps along the secant']
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--accepted',type=Path)
    parser.add_argument('--rejected',type=Path)
    parser.add_argument('--screen-both',action='store_true')
    parser.add_argument('--forecast-steps',type=int,default=1)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
