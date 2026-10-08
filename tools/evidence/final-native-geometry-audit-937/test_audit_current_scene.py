import copy
import importlib.util
import json
from pathlib import Path
import pytest
SPEC=importlib.util.spec_from_file_location("currentaudit",Path(__file__).with_name("audit_current_scene.py"))
a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(a)
RUN=a.E/"native-terminal-cycle-931"
def docs():
    return json.loads((RUN/"invocation.json").read_text()),json.loads((RUN/"run-metadata.json").read_text()),(RUN/"native.log").read_text()
def test_observed_completed_owner_and_true_terminal_are_admitted():
    i,m,t=docs();steps,n,dt=a.validate_documents(i,m,t)
    assert steps == [0,4991,5375,5759,6111,6495,7743,10000]
    assert n==10000 and dt*n==20.000000949949026
@pytest.mark.parametrize("field,value",[("exit_code",1),("source_files_changed_during_run",["changed"]),("loaded_metal_runtime",{"verified":False})])
def test_failed_changed_or_unverified_owner_cannot_supply_anatomy(field,value):
    i,m,t=docs();m[field]=value
    with pytest.raises(ValueError):a.validate_documents(i,m,t)
def test_different_environment_is_rejected():
    i,m,t=docs();m["environment"]["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]="0"
    with pytest.raises(ValueError):a.validate_documents(i,m,t)
@pytest.mark.parametrize("steps",["10001","9999","0,0","1","0,31,63,95,127,159,191,223,255"])
def test_outside_horizon_off_cadence_duplicate_and_over_cap_ids_reject(steps):
    i,m,t=docs()
    i["environment"]["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]=steps
    m["environment"]=copy.deepcopy(i["environment"])
    with pytest.raises(ValueError):a.validate_documents(i,m,t)
@pytest.mark.parametrize("before,after",[("root_assistance=false","root_assistance=true"),
                                       ("simulated_s=20.000000949949026","simulated_s=19.99800094985403"),
                                       ("physiology_body_clock=matched","physiology_body_clock=mismatch")])
def test_finite_exit_is_not_proof_of_unassisted_full_accepted_horizon(before,after):
    i,m,t=docs()
    with pytest.raises(ValueError):a.validate_documents(i,m,t.replace(before,after))
