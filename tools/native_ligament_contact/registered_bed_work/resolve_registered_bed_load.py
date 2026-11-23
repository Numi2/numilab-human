"""Static normal-load test of the native registered posture and its source layer columns."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from fit_bed_section_curve import stress, modulus
from material_context import sha


def material_stretch(pressure, coefficients):
    s = np.ones_like(pressure, dtype=float)
    for _ in range(30):
        s = np.maximum(0.5*s, s + (stress(s, *coefficients)-pressure)/modulus(s, *coefficients))
    if np.max(abs(stress(s, *coefficients)-pressure)) > 2e-7:
        raise ArithmeticError("local constitutive root failed")
    return s


class NormalColumns:
    def __init__(self, columns, coefficients, foundation):
        self.area = columns['area_m2']
        self.height = columns['z_m']
        self.hs, self.hf = columns['skin_m'], columns['fat_m']
        self.coefficients = coefficients
        self.foundation = foundation  # m/Pa of axial foundation compliance
        self.p = np.linspace(0., 50000., 8193)
        self.s, self.f = [material_stretch(self.p, material) for material in coefficients]

    def depth(self, p):
        return (self.foundation*p + self.hs*(1-np.interp(p, self.p, self.s)) +
                self.hf*(1-np.interp(p, self.p, self.f)))

    def response(self, travel):
        d = np.maximum(0., travel-self.height)
        lo = np.zeros_like(d)
        hi = np.full_like(d, self.p[-1])
        if (self.depth(hi) < d).any():
            raise ArithmeticError("compression requires extrapolation")
        for _ in range(31):
            mid = (lo+hi)*0.5
            below = self.depth(mid) < d
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        p = np.where(d>0., (lo+hi)*0.5, 0.)
        if np.max(abs(self.depth(p)-d)) > 1e-8:
            raise ArithmeticError("layer shortening mismatch")
        return p


def assess(columns, material, cap, weight, com, height, E):
    compliance = height/E if E else 0.
    system = NormalColumns(columns, material, compliance)
    soft_stretch = [float(material_stretch(np.array([cap]), m)[0]) for m in material]
    limit = system.height + compliance*cap + system.hs*(1-soft_stretch[0]) + system.hf*(1-soft_stretch[1])
    first = int(np.argmin(limit))
    limit_z = float(limit[first])
    def values(travel):
        p = system.response(travel)
        force = system.area*p
        total = float(force.sum())
        center = np.array([float(force@columns[k]/total) for k in ('x_m','y_m')]) if total else com
        return dict(travel_m=float(travel), total_N=total, contact_area_m2=float(system.area[p>0].sum()),
                    peak_pressure_Pa=float(p.max()), minimum_fat_stretch=float(np.interp(p.max(),system.p,system.f)),
                    bed_compression_fraction=float(p.max()/E if E else 0.),
                    center_of_pressure_xy_m=center.tolist(), applied_moment_about_com_Nm=((center-com)[[1,0]]*np.array([1,-1])*total).tolist(),
                    rows_N=np.bincount(columns['row'].astype(int),weights=force,minlength=38).tolist())
    bound = values(limit_z)
    if abs(bound['peak_pressure_Pa']-cap)>0.02: raise ValueError("first crossing disagrees with material")
    result = dict(foundation_depth_m=height,axial_modulus_Pa=E,first_limited_source_vertex=int(columns['vertex'][first]),
                  first_material_limit=bound)
    if bound['total_N'] >= weight or E:
        hi=limit_z if bound['total_N']>=weight else max(limit_z*1.2,0.075)
        if E:
            while values(hi)['total_N'] < weight:
                hi *= 1.1
        z=brentq(lambda h: values(h)['total_N']-weight,0.,hi,xtol=2e-12)
        balanced=values(z)
        if abs(balanced['total_N']-weight)>2e-5: raise ValueError("unbalanced normal force")
        balanced['stock_remaining']=bool(balanced['bed_compression_fraction']<1.0)
        balanced['within_dense_limit']=bool(balanced['peak_pressure_Pa']<=cap)
        result['axial_force_root']=balanced
    return result


def run(path, curves, capacity, balance):
    cols=np.genfromtxt(path,names=True,delimiter=',')
    coeff=json.loads(curves.read_text())['material']
    for k,v in coeff['material_pins'].items():
        if sha(Path(k))!=v:raise ValueError("unpinned dense material")
    mat=[(m['mu_pa'],m['log_volume_coefficient_pa']) for m in coeff['materials']]
    extrema=json.loads(capacity.read_text())
    with balance.open() as stream:
        target=next(csv.DictReader(stream))
    if int(target["step"])!=0: raise ValueError("missing opening pose")
    cap=extrema['soft_layer_limited_pressure_Pa']
    w=extrema['weight_N']
    if not np.isfinite(cols.view('float64')).all() or len(set(cols['vertex']))!=len(cols) or np.min(cols['area_m2'])<=0 or set(cols['row'])!=set(range(38)):
        raise ValueError("invalid registered columns")
    if abs(float(target['represented_body_mass_kg'])*9.81-w)>0.001:raise ValueError("inconsistent mass")
    cases=[assess(cols,mat,cap,w,np.array([float(target['body_com_'+k+'_m']) for k in 'xy']),h,E)
           for h,E in [(0.,0.),(.05,10000.),(.15,5000.),(.15,10000.),(.15,15000.)]]
    return dict(columns=sha(path),curves=sha(curves),capacity=sha(capacity),mass_reference=sha(balance),
                pressure_ceiling_Pa=cap,gravity_N=w,registered_area_m2=float(cols['area_m2'].sum()),
                positive_area_vertices=len(cols),area_below_one_square_micron_m2=float(cols['area_m2'][cols['area_m2']<1e-12].sum()),
                assumptions="Prepared straight downward translation at fixed orientation. Registered lower-silhouette nodal quadrature, frictionless normal load. Foundation sensitivity uses independent linear axial springs in series with two nonlinear tissue layers. Reported moments use the native opening COM projection. Individual joint balance and dynamic mattress behavior are untested.",
                cases=cases)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('path','curves','capacity','balance'):p.add_argument(arg,type=Path)
    print(json.dumps(run(**vars(p.parse_args())),indent=2,allow_nan=False))
