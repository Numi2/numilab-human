"""Check the condensed force and layer energies against the registered material quadrature."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def field(header,name):
    part=header.split(name+'={',1)[1].split('};',1)[0].replace('std::numeric_limits<std::uint32_t>::max()', '-1')
    return np.array([float(x) for x in part.split(',') if x.strip()])

def audit(footprint, header, controls, trial, target):
    text=header.read_text()
    ids=field(text,'kWholeBodySkinSourceVertices').astype(int)
    skin=field(text,'kWholeBodySkinThicknessM')
    fat=field(text,'kWholeBodyFatThicknessM')
    order={i:j for j,i in enumerate(ids) if i>=0}
    assert len(skin)==len(fat)==len(ids) and len(order)==np.sum(ids>=0)
    foot=table(footprint)
    def values(key,dtype=float):return np.array([row[key] for row in foot],dtype=dtype)
    area=values('tributary_area')
    keep=area>0
    region=values('region',int)[keep]
    sample=np.array([order[v] for v in values('nearest_thickness_vertex',int)[keep]])
    area=area[keep]
    # Uniform approach within a row is the specific assumption tested here.
    depth=skin[sample]/200000+fat[sample]/55000
    old=table(controls/'resting-local-bed-sections.csv')
    new=table(trial/'resting-local-bed-sections.csv')
    assert len(new)==len(old)==len(set(region))==32
    rows=[]
    for a,b in zip(old,new,strict=True):
        r=int(a['contact_index'])
        assert r==int(b['contact_index']) and a['source_geometry_index']==b['source_geometry_index']
        take=region==r
        A=area[take];D=depth[take];sl=skin[sample[take]];fl=fat[sample[take]]
        assert abs(A.sum()-float(b['projected_area_m2']))<1e-13
        for key,t in [('skin_thickness_m',sl),('fat_thickness_m',fl)]:
            assert abs(A@t/A.sum()-float(a[key]))<1e-11 and a[key]==b[key]
        length=1e-5
        force=length*A/D
        pressure=force/A
        skinenergy=.5*np.sum(A*sl*pressure**2/200000)
        fatenergy=.5*np.sum(A*fl*pressure**2/55000)
        total=force.sum()
        cs,cf=[float(b[f'{kind}_compliance_m_per_n']) for kind in ('skin','fat')]
        force_error=abs(length/(cs+cf)/total-1)
        energy_error=max(abs(.5*total**2*cs/skinenergy-1),abs(.5*total**2*cf/fatenergy-1))
        assert max(force_error,energy_error)<2e-7
        oldc=sum(float(a[f'{kind}_compliance_m_per_n']) for kind in ('skin','fat'))
        rows.append(dict(region=r,column_count=len(D),stiffness_ratio=oldc/(cs+cf),
                         force_relative_error=force_error,layer_energy_relative_error=energy_error,
                         first_fat_zero_m=float(np.min(55000*D))))
    result={'native_rows':rows,'tested_columns':len(area),'hashes':{str(p):digest(p) for p in (footprint,header,controls/'resting-local-bed-sections.csv',trial/'resting-local-bed-sections.csv')}}
    for name,folder in [('control',controls),('trial',trial)]:
        q=folder/'resting-support-point-motion.csv'
        if not q.exists():continue
        trace=table(q)
        if len(trace)<32:continue
        worst=0;count=0; witness=None;min_loaded_fat=np.inf
        for p in trace:
            r=int(p['contact_index']);step=int(p['accepted_step']);m=max(0,-float(p['point_world_z_m']))
            ratio=m/rows[r]['first_fat_zero_m']
            if ratio>worst:
                worst=ratio;witness=[step,r]
            if ratio>=1:count+=1
            take=region==r
            min_loaded_fat=min(min_loaded_fat,float(np.min(fat[sample[take]]*(1-m/(55000*depth[take])))))
        result[name]={'sampled_row_times':len(trace),'material_limit_exceedances':count,'peak_fat_compression_fraction':worst,
                      'worst_step_region':witness,'minimum_loaded_fat_m':min_loaded_fat}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('native_rows','hashes')}))
    print('stiffness ratio range',min(r['stiffness_ratio'] for r in rows),max(r['stiffness_ratio'] for r in rows))
if __name__=='__main__':audit(*map(Path,sys.argv[1:]))
