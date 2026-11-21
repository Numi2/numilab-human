"""Weight measured geometric ray paths only at an admissible, force-balanced predictor state."""
import json,sys
from pathlib import Path
import numpy as np
from local_pressure_equilibrium import ColumnLaw
from material_context import sha
def run(shapes,pressure,paths,out):
    g=np.load(shapes);fit=json.loads(pressure.read_text());probes=json.loads(paths.read_text())
    if fit['quadrature']!=shapes.name or probes['queries']!=len(g['xyz']):raise ValueError('Unmatched quadrature')
    rows=probes['traces'];has=np.array([r['bone'] is not None for r in rows])
    covered=np.array([r.get('bone') is not None and not r.get('odd_surface_witnesses') and not r.get('deep_inside_bone') for r in rows])
    def summary(weights):
        w=weights[covered];total=weights.sum()
        if not len(w) or w.sum()==0 or total<=0:raise ValueError('Empty load or area range')
        good=[r for r,c in zip(rows,covered,strict=True) if c]
        def mean(key):return float(w@np.array([r[key] for r in good])/w.sum())
        muscles={}
        for r,p in zip(good,w,strict=True):
            for k,mm in r['contributing_bank_lengths_mm'].items():muscles[k]=muscles.get(k,0.)+p*mm/w.sum()
        return dict(fraction_without_cortical_hit=float(weights[~has].sum()/total),fraction_with_eligible_interval=float(w.sum()/total),
                    mean_deep_to_first_bone_mm=mean('deep_to_bone_mm'),mean_registered_muscle_tendon_union_mm=mean('muscle_or_tendon_union_mm'),
                    mean_interval_outside_those_sources_mm=mean('unaccounted_mm'),top_contributing_banks=[dict(key=k,name=probes['bank_names'][k],mean_mm=v) for k,v in sorted(muscles.items(),key=lambda kv:-kv[1])[:5]])
    result={'source_paths_sha256':sha(paths),'quadrature_sha256':sha(shapes),'pressure_receipt_sha256':sha(pressure),'initial_projected_area':{},'admissible_predictors':{}}
    for mm in (2,5,10,20):
        result['initial_projected_area'][str(mm)+'mm_outer_height']=summary(g['area_m2']*(g['xyz'][:,2]<=mm*.001))
    for mode,run in fit['results'].items():
        last=run['first_material_limit']['last_valid']
        if not last['qualified_within_sampled_material_and_coverage']:raise ValueError('Invalid or incomplete loaded material')
        law=ColumnLaw(g,*fit['materials'],mode,np.array(fit['center_xy_m']))
        q=np.array([last['vertical_approach_m'],*last['gradients']])
        wrench,k,pressure,j,valid=law.evaluate(q)
        if not valid or abs(wrench[0]-last['weight_n'])>1e-5:raise ValueError('Incomplete gravity balance')
        result['admissible_predictors'][mode]=dict(weight_n=wrench[0],**summary(pressure*g['area_m2']))
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
