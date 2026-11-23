"""Recompile the support tables against the new native footprint after root rotation."""
from pathlib import Path
import json,re,sys,subprocess,shutil
import numpy as np
from fit_bed_section_curve import column_force,hermite
from material_context import sha
def build(base,out,case):
    ids=json.loads((base/'identities.json').read_text())
    if any(sha(base/k)!=v for k,v in ids.items()):raise ValueError('stale owner')
    launch=json.loads((case/'launch.json').read_text());curves,=[Path(k) for k in launch['pins'] if k.endswith('/curves.json')]
    mat=json.loads(curves.read_text())['material']['materials']
    props=[(m['mu_pa'],m['log_volume_coefficient_pa']) for m in mat]
    g=np.genfromtxt(case/'resting-local-bed-footprint.csv',delimiter=',',names=True)
    sections=np.genfromtxt(case/'resting-local-bed-sections.csv',delimiter=',',names=True)
    if set(g['row'])!=set(range(38)) or list(sections['contact_index'])!=list(range(38)):raise ValueError('incomplete source rows')
    data=[]; inc=[]; checks=[]; errors=np.zeros(2)
    for i,r in enumerate(sections):
        row=g[g['row']==i]
        area,sk,ft=[row[k] for k in ['area_m2','skin_m','fat_m']]
        if min(np.min(x) for x in (area,sk,ft))<=0 or abs(area.sum()/r['projected_area_m2']-1)>1e-7:raise ValueError('invalid material stock')
        h=np.float32(.95*np.min(sk+ft))/np.float32(128)
        F,K=column_force(np.arange(129)*h,area,sk,ft,props);v=np.stack([F,K],axis=1).astype('f4')
        checks.append(1./K[0]);inc.append(h);data.extend(v)
        for u in [.13,.5,.87]:
            direct=column_force((np.arange(128)+u)*h,area,sk,ft,props);pred=hermite(v[:-1],v[1:],h,np.full(128,u))
            errors=np.maximum(errors,[np.max(abs(a-b)/np.maximum(abs(a),1e-7)) for a,b in zip(direct,pred)])
    if np.max(errors)>.002:raise ValueError(('legacy interpolation',errors))
    subprocess.run(['cp','-cR',base,out],check=True)
    path=out/'numilab_human_myosim_visual_probe.mm';s=path.read_text()
    for name,items in [('double initialCompliance',[float(x).hex() for x in checks]),('unsigned sourceIds',[str(int(r['source_geometry_index'])) for r in sections])]:
        s,count=re.subn(r'(constexpr '+name+r'\[\] = \{)[^}]+\}',lambda m:m[1]+','.join(items)+'}',s)
        if count!=1:raise ValueError('unknown source table binding')
    path.write_text(s)
    def lit(a):return float(a).hex()+'f'
    (out/'NumiHumanBedForceCurves.metalinc').write_text('constant float2 forceAndModulus[] = {\n'+',\n'.join('float2('+lit(a)+','+lit(b)+')' for a,b in data)+'\n};\nconstant float depthIncrement[] = {'+','.join(map(lit,inc))+'};\n')
    shutil.copy2(out.parent/'NumiHumanFoamFoundation.metalinc',out)
    cwd=Path(launch['environment']['DYLD_LIBRARY_PATH']).parent
    for name in ['shader-build-commands.json','build-commands.json']:
        cmd=json.loads((base/name).read_text());old=Path(cmd[0][cmd[0].index('-c')+1]).parent
        cmds=[[v.replace(str(old)+'/',str(out)+'/').replace(str(base)+'/',str(out)+'/') for v in c] for c in cmd]
        for c in cmds:subprocess.run(c,cwd=cwd,check=True)
        (out/name).write_text(json.dumps(cmds,indent=2)+'\n')
    (out/'identities.json').write_text(json.dumps({k:sha(out/k) for k in ids},indent=2)+'\n')
    (out.parent/'table-check.json').write_text(json.dumps(dict(max_relative_force_and_tangent_error=errors.tolist(),source=sha(case/'resting-local-bed-footprint.csv'),sections=sha(case/'resting-local-bed-sections.csv')),indent=2)+'\n')
if __name__=='__main__':build(*(Path(x).resolve() for x in sys.argv[1:]))
