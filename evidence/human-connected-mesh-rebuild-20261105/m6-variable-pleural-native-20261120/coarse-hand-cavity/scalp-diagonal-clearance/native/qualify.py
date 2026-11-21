from pathlib import Path
from collections import Counter
import json
run = Path("native/root-128-01")
new = json.loads((run/"bone-endpoints.json").read_text())
expected = json.loads(Path("exact-02/bone-witnesses.json").read_text())
a = json.loads((run/"verified-shoulders.json").read_text())
c = json.loads((run/"verified-cells.json").read_text())
m = json.loads((run/"verified-mass.json").read_text())
report = json.loads((run/"comparison.json").read_text())
def keys(source):
    return {(role,tuple(sorted(face)),body,bone,triangle)
            for role,_,face,body,bone,triangle in source["witnesses"]}
def counts(witnesses):
    return {phase:len(keys(source)) for phase,source in witnesses.items() if phase in {"initial","terminal"}}
for phase in ("initial","terminal"):
    assert keys(new[phase]) == keys(expected[phase]), phase
    q = c["endpoints"][phase]
    assert (q["inverted_or_degenerate_cells"] == 0 and q["cells_below_quality_floor"] == 0
            and not q["shell_crossing_pairs"])
    assert report["crossings"][phase]["candidate_pairs"] < report["crossings"][phase]["control_pairs"]
assert m["changed_typed_cell_rows"] == 0
assert m["maximum_cell_mass_error_kg"] == 0.
assert all(not side["unresolved_source_bone_hits"] and not side["barycentric_path_hits"]
           for pose in a["captures"] for side in pose["shoulders"])
assert [p["accepted_step"] for p in a["captures"]] == [1,32,64,96,128]
terminal = (run / "terminal.log").read_text()
assert ("resting_fascia_fem_terminal_audit=passed " in terminal and
        "registered_ligament_force_rows=34 " in terminal)
assert report["candidate"]["pleural_pa_range"][0] < report["candidate"]["pleural_pa_range"][1]
data = {"physical_status":"native 128-root release accepted", "motion_time_s":report["candidate"]["simulated_s"],
        "native_geometry_matches_predeclared_witnesses": {phase:True for phase in ("initial","terminal")},
        "bone_pair_counts":counts(new), "retained_ligament_rows":34, "shoulder_capture_steps":[1,32,64,96,128],
        "bed_mean_force_N":report["candidate"]["last_accepted_bed_force_N"], "used_artificial_support":False}
(run / "qualification.json").write_text(json.dumps(data,indent=2)+"\n")
print(data)
