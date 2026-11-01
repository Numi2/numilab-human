#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, resource, signal, sys, time, traceback
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OWNER=ROOT/'quality-endpoint-collapse-001/postquotient_cavity_remesh_883h_seed_star_cap48.py'
OWNER_SHA='310c2bc2463e98abfa00c9518f319973d976c3c7de5348e5f05b6e12033bec26'
PARENT_SHA='0b4cc4d9e4ef95d4d64f33bb08cf32da3d7bb3246b7fa853dfc8447e7d46f16b'
DIRECTIONS=((33005,33006),(33005,33004))
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def cv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def peak_rss_bytes():
    x=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return x if sys.platform=='darwin' else x*1024
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--solve-seconds',type=float,default=10.)
    a=ap.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    if sha(OWNER)!=OWNER_SHA:raise RuntimeError('883h producer pin mismatch')
    if sha(a.state)!=PARENT_SHA:raise RuntimeError('qualified post-883g parent state pin mismatch')
    spec=importlib.util.spec_from_file_location('evaluate_883h_short_edges',OWNER)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    if (m.MAX_CELLS,m.MAX_POINTS,m.MAX_CANDIDATE_COMBINATIONS)!=(128,48,200_000):
        raise RuntimeError('883h bounded construction policy differs')
    original=m.solve_role
    attempts=[]
    def capture_factory(records):
        def capture(*args,**kwargs):
            r=original(*args,**kwargs)
            sel=r.get('cavity_selection',{})
            sol=r.get('solver',{})
            records.append({'role':r['role'],'selection_mode':sel.get('mode'),
                'selection_policy':sel.get('policy'),'seed_cells':sel.get('complete_endpoint_star_cell_count'),
                'ring_cells':sel.get('ring_candidate_cell_count'),'selected_cells':sel.get('selected_cell_count'),
                'solver_points_after_drop':sel.get('selected_solver_point_count_after_drop'),
                'candidate_combinations_upper_bound':sel.get('candidate_combinations_upper_bound'),
                'candidate_combinations_cap':sel.get('candidate_combinations_cap'),
                'candidate_count_total_combinations':sol.get('candidate_count_total_combinations'),
                'candidate_count_positive_exact_float32':sol.get('candidate_count_positive_exact_float32'),
                'fallback_reasons':sel.get('fallback_reasons',[]),'native_bad_before':r['native_bad_before'],
                'native_bad_after':r['native_bad_after'],'log_deficit_before':r['log_deficit_before'],
                'log_deficit_after':r['log_deficit_after'],'exact_det6_delta':str(r['exact_det6_delta']),
                'solver_status':sol.get('status'),'solver_elapsed_s':sol.get('elapsed_s')})
            return r
        return capture
    t0=time.monotonic();rss0=peak_rss_bytes()
    signal.signal(signal.SIGALRM,lambda signum,frame: (_ for _ in ()).throw(TimeoutError('120s evaluation wall limit reached')))
    signal.setitimer(signal.ITIMER_REAL,120.0)
    try:
        for drop,keep in DIRECTIONS:
            if time.monotonic()-t0>=120.0:
                attempts.append({'drop':drop,'keep':keep,'status':'NOT_RUN_WALL_LIMIT','geometry_written':False})
                continue
            edge=tuple(sorted((drop,keep)))
            cavities=[]
            od=out/f'edge-{edge[0]}-{edge[1]}-drop-{drop}-keep-{keep}'
            direction_start=time.monotonic()
            m.solve_role=capture_factory(cavities)
            m.run_one._evaluation_only=True
            try:
                result=m.run_one(a.state,edge,drop,keep,od,fixture=False,
                    solve_seconds=min(10.,a.solve_seconds),source_role='I')
                attempts.append({'edge':list(edge),'drop':drop,'keep':keep,
                    'status':result['status'],'source_edge_length_m':result.get('source_edge_length_m'),
                    'quality_objective':result.get('quality_objective'),
                    'role_metrics':result.get('role_metrics'),'cavities':cavities,
                    'direction_elapsed_s':time.monotonic()-direction_start,'geometry_written':False})
            except Exception as exc:
                attempts.append({'edge':list(edge),'drop':drop,'keep':keep,'status':'REJECTED',
                    'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
                    'cavities':cavities,'direction_elapsed_s':time.monotonic()-direction_start,'geometry_written':False})
            finally:
                m.run_one._evaluation_only=False
                m.solve_role=original
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.0)
        m.run_one._evaluation_only=False
        m.solve_role=original
    elapsed=time.monotonic()-t0
    report={'schema':'h706.883h.current-short-I-edge-evaluation-only.v1',
        'status':'COMPLETED_EVALUATION_ONLY' if elapsed<=120.0 else 'WALL_LIMIT_REACHED',
        'producer':{'path':str(OWNER),'sha256':OWNER_SHA},
        'parent_state':{'path':str(a.state.resolve()),'sha256':sha(a.state)},
        'directions':[[int(d),int(k)] for d,k in DIRECTIONS],
        'attempt_count':len(attempts),'attempts':attempts,'elapsed_s':elapsed,'wall_limit_s':120.,
        'max_solve_seconds_per_cavity':min(10.,a.solve_seconds),
        'construction_caps':{'cells':m.MAX_CELLS,'solver_points_after_drop':m.MAX_POINTS,
            'candidate_combinations':m.MAX_CANDIDATE_COMBINATIONS,
            'max_combinations_at_point_cap':math.comb(m.MAX_POINTS,4)},
        'peak_rss_bytes_start':rss0,'peak_rss_bytes_end':peak_rss_bytes(),
        'candidate_serialization':'NOT_RUN','independent_volume_replay':'NOT_RUN',
        'external_775_805_all185':'NOT_RUN','physical_admission':False,
        'native_steps':0,'mass_mapping':'PENDING'}
    p=out/'report.json';p.write_text(json.dumps(report,indent=2,sort_keys=True,default=cv)+'\n')
    print(json.dumps({'report':str(p),'status':report['status'],'attempts':len(attempts),'elapsed_s':elapsed}))
if __name__=='__main__':main()
