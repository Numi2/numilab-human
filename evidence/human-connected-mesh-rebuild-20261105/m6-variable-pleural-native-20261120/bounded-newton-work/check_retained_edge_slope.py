"""Independently check the calculated barrier slope on the retained witness chord."""
from pathlib import Path
import json
import math

import numpy as np

root = Path.cwd()
retained = json.loads((root/"late-barrier-witness.json").read_text())
c = retained["unmodified_detail"]["contacts"][0]

def words(high,low):
    return np.asarray(high,dtype=np.float32).astype(np.float64)[:,:3] + np.asarray(low,dtype=np.float32).astype(np.float64)[:,:3]

start = words(c["potential_feature_start_high"], c["potential_feature_start_low"])
end = words(c["potential_feature_end_high"], c["potential_feature_end_low"])
p = start
v = end-start
e = p[1]-p[0]
f = p[3]-p[2]
dv = v[1]-v[0]
dw = v[3]-v[2]
r = p[0]-p[2]
s,t = np.linalg.lstsq(np.column_stack((e,-f)),-r,rcond=None)[0]
assert 0 < s < 1 and 0 < t < 1
rel = r+s*e-t*f
distance = np.linalg.norm(rel)
distance_derivative = rel@(v[0]-v[2]+s*dv-t*dw)/distance
thickness = c["barrier_parameters"][2]
k = c["barrier_parameters"][3]
sine_squared = np.cross(e,f)@np.cross(e,f)/(e@e)/(f@f)
ratio = min(sine_squared/1.e-3,1.)
m = ratio*(2.-ratio)
normal = np.cross(e,f)
normal_v = np.cross(dv,f)+np.cross(e,dw)
angular_speed = (2.*(normal@normal_v)/(e@e)/(f@f) -
                 sine_squared*(2.*(e@dv)/(e@e)+2.*(f@dw)/(f@f)))
m_derivative = 2.*(1.-ratio)*angular_speed/1.e-3 if ratio<1. else 0.
q = distance**2-thickness**2
l = math.log1p(q/thickness**2)
radial = -2.*distance*k*(2.*q*l+q*q/distance**2)
energy = -k*q*q*l
predicted = m*radial*distance_derivative+m_derivative*energy

def fit_energy(points):
    edge=points[1]-points[0]
    other=points[3]-points[2]
    d=points[0]-points[2]
    u,w=np.linalg.lstsq(np.column_stack((edge,-other)),-d,rcond=None)[0]
    assert 0<u<1 and 0<w<1
    d=d+u*edge-w*other
    sq=d@d-thickness**2
    cross=np.cross(edge,other)
    mpart=min((cross@cross)/(edge@edge)/(other@other)/1.e-3,1.)
    return -k*sq*sq*math.log1p(sq/thickness**2)*mpart*(2.-mpart)

h=0.01
centered=(fit_energy(p+h*v)-fit_energy(p-h*v))/(2.*h)
assert math.isclose(centered,predicted,rel_tol=1.e-2)
report={
    "method":"Float64 replay of exact stored high/low binary32 geometry along the certified full rounded chord",
    "start_separation_m":float(distance),
    "start_energy_j":float(m*energy),
    "rounded_chord_energy_derivative_j":float(predicted),
    "centered_difference_j":float(centered),
    "stored_force_row_slope_j":retained["stored_slope_j"],
    "regional_replay_only":True,
}
(root/"late-edge-slope-check.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
