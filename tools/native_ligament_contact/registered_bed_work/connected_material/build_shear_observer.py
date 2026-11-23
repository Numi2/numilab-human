"""Observe the three physical contact Jacobians at the first step for source recruitment."""
import json
import subprocess
import shutil
import sys
from pathlib import Path
from export_contact_columns import change
from run_bed_work import digest

def prepare(observer,active,source,out):
    ids=json.loads((observer/'identities.json').read_text())
    native=json.loads((active/'identities.json').read_text())
    if out.exists() or any(digest(observer/k)!=v for k,v in ids.items()) or any(digest(active/k)!=v for k,v in native.items()):
        raise ValueError('Observer source or active model changed')
    for k in ('NumiHumanStand.metal','NumiHumanStandSolve.metalinc','NumiHumanBedForceCurves.metalinc','NumiHumanRestingVisual.hpp','NumiHumanFoamFoundation.metalinc','numilab_human_myosim_visual_probe.mm'):
        if digest(active/k)!=digest(observer/k):
            raise ValueError('Unmatched observation '+k)
    subprocess.run(['cp','-cR',observer,out],check=True)
    shutil.copy2(active/'NumiHumanRegisteredRecruitment.inc',out/'NumiHumanRegisteredRecruitment.inc')
    p=out/'NumiHumanResolvedBed.metalinc'
    s=change(p.read_text(),'if(step==0u && axis==2u)\n        openingForce[row*(nv+1u)+component] = component==nv ? F : action;',
             'if(step==0u)\n        openingForce[(3u*row+axis)*(nv+1u)+component] = component==nv ? F : action;')
    p.write_text(s)
    p=out/'NumiHumanResolvedBed.hpp'
    s=change(p.read_text(),'std::vector<float>(regionCount_*openingStride_,NAN)','std::vector<float>(3u*regionCount_*openingStride_,NAN)')
    s=change(s,'table<<std::setprecision(12)<<"row";','table<<std::setprecision(12)<<"row,axis";')
    s=change(s,'for(size_t r=0;r<regionCount_;++r) {\n                table<<r;',
                'for(size_t r=0;r<3u*regionCount_;++r) {\n                table<<r/3u<<\',\'<<r%3u;')
    p.write_text(s)
    commands=[[v.replace(str(observer)+'/',str(out)+'/') for v in c] for c in json.loads((observer/'build-commands.json').read_text())]
    cwd=Path(json.loads((source/'launch.json').read_text())['environment']['DYLD_LIBRARY_PATH']).parent
    for c in commands:subprocess.run(c,cwd=cwd,check=True)
    (out/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    (out/'identities.json').write_text(json.dumps({k:digest(out/k) for k in ids},indent=2)+'\n')
if __name__=='__main__':prepare(*(Path(s).resolve() for s in sys.argv[1:]))
