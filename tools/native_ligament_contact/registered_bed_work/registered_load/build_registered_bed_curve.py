"""Compile the existing finite-strain bed law for the current registered native contact rows."""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from compile_connected_joint import digest
from fit_bed_section_curve import column_force, hermite


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def physical_curves(trace, source, reference):
    footprint = trace/'resting-local-bed-footprint.csv'
    if digest(footprint) != digest(reference/footprint.name):
        raise ValueError('Unmatched anatomical contact footprint')
    columns = read_rows(footprint)
    sections = read_rows(trace/'resting-local-bed-sections.csv')
    points = read_rows(trace/'resting-support-selected-skin-witness.csv')
    if not columns or len({r['vertex'] for r in columns}) != len(columns):
        raise ValueError('Missing or duplicate footprint vertices')
    anchors = {}
    for r in points:
        key = int(r['region'])
        value = (int(r['source_geometry_index']), int(r['point_query_index']))
        if anchors.setdefault(key, value) != value or int(r['region_error']) or int(r['global_error']):
            raise ValueError('Ambiguous registered query')
    if set(anchors) != set(range(len(sections))) or set(int(r['row']) for r in columns) != set(anchors):
        raise ValueError('Incomplete registered coverage')
    first = anchors[0][1]
    materials = json.loads(source.read_bytes())['materials']
    props = [(materials[str(i)]['registered']['source_mu_pa'],
              materials[str(i)]['registered']['source_logJ_coefficient_pa']) for i in (15,16)]
    from fit_bed_section_curve import stress
    pressure_limit = min(stress(0.35, *m) for m in props)
    limit_stretches = [brentq(lambda s: stress(s,*m)-pressure_limit,0.349999,1.0,xtol=1e-14) for m in props]
    curves, metadata = [], []
    peak = np.zeros(2)
    for i, section in enumerate(sections):
        geom, point = anchors[i]
        rows = [r for r in columns if int(r['row']) == i]
        if (point != first+i or int(section['contact_index']) != i or
                int(section['source_geometry_index']) != geom or
                any(int(r['source_geometry_index']) != geom for r in rows)):
            raise ValueError('Contact identifiers disagree')
        area,skin,fat = [np.array([float(r[k]) for r in rows]) for k in ('area_m2','skin_m','fat_m')]
        if not np.all(np.isfinite([area,skin,fat])) or np.min([area,skin,fat]) <= 0 or abs(area.sum()/float(section['projected_area_m2'])-1)>1e-7:
            raise ValueError('Invalid registered column geometry')
        domain = np.min(skin*(1-limit_stretches[0]) + fat*(1-limit_stretches[1]))
        h = np.nextafter(np.float32(domain/128),np.float32(0))
        depths = np.arange(129)*float(h)
        f,k = column_force(depths,area,skin,fat,props)
        vals = np.stack([f,k],axis=1).astype(np.float32)
        compliance = float(section['skin_compliance_m_per_n'])+float(section['fat_compliance_m_per_n'])
        if not np.all(np.isfinite(vals)) or min(k)<=0 or abs(compliance*k[0]-1)>1e-5 or float(depths[-1])>domain:
            raise ValueError(('Opening material or thickness mismatch',i,compliance*k[0]))
        for offset in (0.125,0.5,0.875):
            direct = column_force((np.arange(128)+offset)*float(h),area,skin,fat,props)
            gpu = np.array(hermite(vals[:-1],vals[1:],h,np.full(128,offset)))
            err = np.max(abs(gpu-direct)/np.maximum(abs(np.array(direct)),1e-7),axis=1)
            if not np.all(np.isfinite(err)) or np.any(err>.002) or np.min(gpu)<=0:
                raise ValueError(('GPU curve disagrees with registered law',i,err))
            peak = np.maximum(peak,err)
        curves.append(vals)
        metadata.append(dict(source=geom,query=point,columns=len(rows),area_m2=float(sum(area)),
            increment_m=float(h),limit_m=float(depths[-1]),compliance=1/k[0],capacity_N=f[-1]))
    return curves, metadata, dict(force_tangent_relative_errors=peak.tolist(),material=digest(source),
        footprint=digest(footprint),sections=digest(trace/'resting-local-bed-sections.csv'),
        points=digest(trace/'resting-support-selected-skin-witness.csv'),
        material_J_min=.35,column_pressure_limit_Pa=pressure_limit,limiting_stretches=limit_stretches)


def build(linear, trace, material, control, shader, finite, dest):
    dest.mkdir()
    curves, rows, result = physical_curves(trace,material,control)
    def literal(value):
        return float(value).hex()+'f'
    table = 'constant float2 forceAndModulus[] = {\n'+',\n'.join(
        'float2('+literal(f)+','+literal(k)+')' for curve in curves for f,k in curve)+'\n};\n'
    table += 'constant float depthIncrement[] = {'+','.join(literal(r['increment_m']) for r in rows)+'};\n'
    inc = 'NumiHumanBedForceCurves.metalinc'
    (dest/inc).write_text(table)
    stand = 'NumiHumanStand.metal'
    baseline = (shader/stand).read_text()
    previous = (finite/stand).read_text()
    marker='inline StandContactNormalLaw standContactNormalLaw('
    end='inline float3 standContactPlaneNormal('
    def law(s):
        if s.count(marker)!=1 or s.count(end)!=1:raise ValueError('Unexpected contact solver')
        return s[s.index(marker):s.index(end)]
    old=law(previous)
    if old.count('628u')!=2 or old.count('666u')!=1:raise ValueError('Unexpected finite law inputs')
    new=old.replace('628u',str(rows[0]['query'])+'u').replace('666u',str(rows[-1]['query']+1)+'u')
    (dest/stand).write_text(baseline.replace(law(baseline),'#include "'+inc+'"\n\n'+new))
    for f in linear.iterdir():
        if f.suffix in ('.hpp','.metalinc') or f.name in ('human_respiration.metal','human_resting_runtime.hpp'):
            (dest/f.name).symlink_to(f.resolve())
    solve='NumiHumanStandSolve.metalinc'
    (dest/solve).write_bytes((shader/solve).read_bytes())
    (dest/'metalrobo').mkdir()
    (dest/'metalrobo/HumanRestingSkin.metallib').symlink_to((linear/'metalrobo/HumanRestingSkin.metallib').resolve())
    cfg=json.loads((control/'launch.json').read_bytes())
    cwd=Path(cfg['environment']['DYLD_LIBRARY_PATH']).parent
    commands=json.loads((shader/'shader-build-commands.json').read_bytes())
    metal='metalrobo/MetalRobo.metallib'
    check=dest/'control.metallib'
    subprocess.run(commands[1][:-1]+[str(check)],cwd=cwd,check=True)
    if digest(check)!=digest(linear/metal):raise ValueError('Unmatched control shader objects')
    check.unlink()
    adjusted=[[x.replace(str(shader)+'/',str(dest)+'/') for x in cmd] for cmd in commands]
    if str(dest/stand) not in adjusted[0] or adjusted[1][-1]!=str(dest/metal):raise ValueError('Unexpected Metal output')
    host='numilab_human_myosim_visual_probe.mm'
    base=(linear/host).read_text()
    anchor='            row.planePoint.w = section.skinComplianceMPerN + section.fatComplianceMPerN;'
    if base.count(anchor)!=1:raise ValueError('Unexpected host owner')
    guards='            constexpr unsigned sourceIds[] = {'+','.join(str(r['source']) for r in rows)+'};\n'
    guards+='            constexpr double openingCompliance[] = {'+','.join(float(r['compliance']).hex() for r in rows)+'};\n'
    guards+='            require(localBedSections.size() == '+str(len(rows))+'u && row.pointQueryIndex == '+str(rows[0]['query'])+'u+i &&\n'
    guards+='                    row.sourceGeometryIndex == sourceIds[i] && std::abs((section.skinComplianceMPerN+section.fatComplianceMPerN)/openingCompliance[i]-1.0)<1e-5,\n'
    guards+='                    "registered finite-strain contact table disagrees with current support");\n'
    (dest/host).write_text(base.replace(anchor,guards+anchor))
    build_cfg=json.loads((linear/'registered-sections.json').read_bytes())
    adjusted += [[x.replace(str(linear)+'/',str(dest)+'/') for x in cmd] for cmd in build_cfg['commands']]
    with (dest/'build.log').open('w') as log:
        for cmd in adjusted:
            subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
    result.update(rows=rows,owner_shader=digest(shader/stand),prior_finite_shader=digest(finite/stand),
                  commands=adjusted,binary=digest(dest/'numi-human-native'),gpu=digest(dest/metal),
                  control_gpu=digest(linear/metal))
    (dest/'curves.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('qualified material rows',len(rows),'errors',result['force_tangent_relative_errors'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('linear','trace','material','control','shader','finite','dest'):
        p.add_argument(k,type=lambda s:Path(s).resolve())
    build(**vars(p.parse_args()))
