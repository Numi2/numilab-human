from pathlib import Path
import csv,functools,hashlib,importlib.util,json,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
N=W/'unchanged-mesh-repeat-376';R=W/'flat-cap-native-365'
sys.path.insert(0,str(W))
import compare_remeshed_native_310 as physical
import compare_matched_deformation_339 as matched
spec=importlib.util.spec_from_file_location('controls','/Users/numi/Work/human-checkpoint-owner-20261105/tools/compare_human_native_mesh_trials.py')
controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
cp=json.loads((N/'install-motion.plan.json').read_text());rp=json.loads((R/'install-motion.plan.json').read_text())
diff=controls.control_differences(cp,rp)
assert diff==[],diff
compatibility=dict(matched=True,differences=diff,qualification='Matched runtime and numerical controls. Existing captured-force and regional-strain engineering screen; no material-energy or sustained-motion qualification.')
raw_capture=physical.capture
physical.capture=functools.lru_cache(maxsize=2)(raw_capture)
ref=R/'install-motion.stdout.log.gz';candidate=N/'install-motion.stdout.log.gz'
# Reuse exactly the same two parsed endpoint arrays for both retained analyses.
a,b=physical.capture(ref),physical.capture(candidate)
result=physical.compare(ref,candidate);result['solver_control_comparison']=compatibility
(N/'endpoint-force-strain-diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
deformation=matched.compare_matched(a,b);deformation['solver_control_comparison']=compatibility
deformation['inputs']={k:dict(path=str(p),sha256=physical.sha(p)) for k,p in [('reference',ref),('candidate',candidate)]}
(N/'endpoint-matched-field-diagnostic.json').write_text(json.dumps(deformation,indent=2,allow_nan=False)+'\n')
force=[v['relative_resultant_difference_over_reference_nodal_l1'] for region in result['regional_force_channels'].values() for v in region.values()]
strain=[v for region in result['regional_strain'].values() for v in region['relative_change'].values()]
summary=dict(solver_control_comparison=compatibility,force_metrics=len(force),force_metrics_over_1percent=sum(v is None or v>.01 for v in force),strain_metrics=len(strain),strain_metrics_over_5percent=sum(v is None or v>.05 for v in strain),maximum_force_metric=max(v for v in force if v is not None),maximum_strain_metric=max(v for v in strain if v is not None),total_fem_mass_kg=result['total_fem_mass_kg'],matched_field=deformation['identical_cells'],physical_fidelity_qualification=False,source_owners={str(p):physical.sha(p) for p in [Path(__file__),Path(physical.__file__),Path(matched.__file__)]})
(N/'endpoint-diagnostic-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2),flush=True)
