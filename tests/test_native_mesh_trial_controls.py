"""Prevent incomparable native runs from being presented as matched solver trials."""
from copy import deepcopy
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('mesh_trials',Path(__file__).resolve().parents[1]/'tools/compare_human_native_mesh_trials.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)

def fixture():
 return {'stages':[{'argv':['/native','/rigid','/muscle','/bones','/control/out','--muscle-step-count','32','--muscle-step-seconds','0.000015625'],
                    'environment':{'NM_MATTER_FEM_KRYLOV_RESTART':'16','BRAIN_ENABLED':'1','TRACE_PATH':'/control/out/trace','NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_ASSET':'/old-geometry'}}],
         'native_binary':{'sha256':'native-code'},'runtime_dependencies':{'/matter.metallib':{'sha256':'shader-code'}}}

def test_geometry_and_output_changes_are_allowed():
 a=fixture();b=deepcopy(a);b['stages'][0]['argv'][4]='/candidate/out'
 b['stages'][0]['environment']['TRACE_PATH']='/candidate/out/trace'
 b['stages'][0]['environment']['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_ASSET']='/new-geometry'
 assert owner.control_differences(a,b)==[]

def test_linear_budget_mismatch_is_rejected():
 a=fixture();b=deepcopy(a);b['stages'][0]['environment']['NM_MATTER_FEM_KRYLOV_RESTART']='32'
 assert 'environment:NM_MATTER_FEM_KRYLOV_RESTART' in owner.control_differences(a,b)

def test_time_step_mismatch_is_rejected():
 a=fixture();b=deepcopy(a);b['stages'][0]['argv'][-1]='0.002'
 assert 'argv' in owner.control_differences(a,b)

def test_runtime_mismatch_is_rejected():
 a=fixture();b=deepcopy(a);b['native_binary']['sha256']='different-code'
 b['runtime_dependencies']['/matter.metallib']['sha256']='different-shader'
 assert set(owner.control_differences(a,b))=={'native_binary','runtime_dependencies'}

def test_controller_change_is_rejected():
 a=fixture();b=deepcopy(a);b['stages'][0]['environment']['BRAIN_ENABLED']='0'
 assert 'environment:BRAIN_ENABLED' in owner.control_differences(a,b)
