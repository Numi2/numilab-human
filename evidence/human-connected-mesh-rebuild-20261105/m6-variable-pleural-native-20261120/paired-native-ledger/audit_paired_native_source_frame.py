"""Compare the archived carrier frame to the compiled material-reference frame."""
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
previous = root.parent / "resting-clearance-and-atrium-016"
sys.path.insert(0,str(previous))
from screen_paired_mass_routes import COMPILER, paired_reference_carriers, sha

earlier = root.parent / "resting-clearance-and-atrium-010"
sys.path.insert(0,str(earlier))
from rebind_current_skin import read_literal

state_file = root/"complete-fine-p1/current-paired-volume-state.json"
state = json.loads(state_file.read_text())
for pin in state["source330_identity"].values():
    if sha(Path(pin["path"])) != pin["sha256"]:
        raise ValueError("Unverified source frame input")
inputs = {k:Path(v["path"]) for k,v in state["source330_identity"].items()}
old = np.loadtxt(inputs["source330_nodes"],delimiter=",",skiprows=1)
tet_table = np.loadtxt(inputs["source330_tets"],delimiter=",",skiprows=1)
archive = old[:,5:8]
rest, tets, mass, previous_report = paired_reference_carriers(
    COMPILER, inputs["source330_nodes"],inputs["source330_tets"])
with np.load(earlier/"paired-skin-fat-refit-001/paired-geometry-and-mass.npz",allow_pickle=False) as paired:
    corrected_geometry = paired["points_world_m"].reshape(-1,3)
apps = Path.home()/"Work/skin-fat-native-integration-209/apps"
coarse = apps/"NumiHumanWholeBodySkinReduction.hpp"
candidate = apps/"NumiHumanFinal128CandidateRegistration.hpp"
def read(path,key):
    return read_literal(path,key).reshape(-1,3)
historical = np.vstack((
    read(candidate,"kWholeBodySkinCandidateReferenceWorldPoints")
       + read(coarse,"kWholeBodySkinMechanicalWorldDeltas"),
    read(coarse,"kWholeBodySkinInnerWorld"),
    read(coarse,"kWholeBodyFatDeepWorld"),
))
def fit(x,y):
    xi, yi = x.mean(axis=0), y.mean(axis=0)
    u, _, vt = np.linalg.svd((x-xi).T @ (y-yi))
    if np.linalg.det(u @ vt) < 0.0:
        u[:,-1] *= -1.
    r = u @ vt
    t = yi - xi @ r
    return r,t,np.linalg.norm(x @ r + t - y,axis=1)
density = np.select((tet_table[:,1]==15,tet_table[:,1]==16),(1000.,900.))
def certificate(points):
    e = points[tets]
    d = e[:,1:] - e[:,:1]
    volumes = np.einsum('ij,ij->i',d[:,0],np.cross(d[:,1],d[:,2]))/6.
    masses = volumes*density
    report={"inverted": int(np.count_nonzero(volumes <= 0)),
            "volume_m3":float(volumes.sum()),"mass_at_reference_density_kg":float(masses.sum()),
            "max_abs_replay_delta_kg":float(np.abs(masses-mass).max())}
    return report
rotation,translation,rest_error = fit(historical,archive)
r2,t2,current_error = fit(historical,old[:,2:5])
rebuilt = corrected_geometry @ rotation + translation
report = {"state_sha256":sha(state_file),"nodes":len(archive),"cells":len(tets),
          "historical_world_to_captured_rest_R":rotation.tolist(),
          "historical_world_to_captured_rest_t":translation.tolist(),
          "historical_rest_alignment_max_m":float(rest_error.max()),
          "historical_current_alignment_max_m":float(current_error.max()),
          "historical_rest":certificate(archive),
          "paired_raw":certificate(corrected_geometry),
          "paired_carrier_registration_frame":certificate(rest),
          "paired_single_rigid_rest_frame":certificate(rebuilt),
          "max_paired_frame_position_difference_m":float(np.linalg.norm(rebuilt-rest,axis=1).max()),
          "native_admitted":False}
(root/"paired-native-source-frame-audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
