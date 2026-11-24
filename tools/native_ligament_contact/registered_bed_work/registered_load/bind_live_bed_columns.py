"""Bind the existing per-vertex Metal foundation to the registered pelvic and forearm support scene."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from compile_connected_joint import digest
from build_surface_audit_order import correct_order


def replace(text,old,new):
    if text.count(old)!=1:
        raise ValueError('Unknown native owner near '+old[:80])
    return text.replace(old,new)


def excerpt(text,begin,end):
    return text[text.index(begin):text.index(end,text.index(begin))]


def build(owner,finite,prior,control,dest):
    dest.mkdir();(dest/'metalrobo').mkdir()
    cfg=json.loads((control/'launch.json').read_bytes())
    def save(name,content):
        (dest/name).write_text(content)
    def copy(src,name):
        shutil.copyfile(src/name,dest/name)
    vis='NumiHumanRestingVisual.hpp';stand='NumiHumanStand.metal';host='numilab_human_myosim_visual_probe.mm'
    geometry='NumiHumanRestingSupportGeometry.hpp';footprint='NumiHumanBedFootprint.hpp'
    overrides={vis,stand,host,geometry,footprint,'human_respiration.metal','NumiHumanResolvedBed.hpp'}
    for f in finite.iterdir():
        if f.name not in overrides and f.is_file() and f.suffix in ('.hpp','.metalinc'):
            (dest/f.name).symlink_to(f.resolve())
    for f in ('NumiHumanResolvedBed.metalinc','NumiHumanBedPressureTable.metalinc',footprint):
        copy(prior,f)
    n='NumiHumanResolvedBed.hpp'
    h=(prior/n).read_text();save(n,replace(h,'count==38 &&','(count==38 || count==44) &&'))
    # Validate the copied closure table against the two current material definitions.
    import re
    import numpy as np
    from compare_support_materials import compressed_stretch
    m=json.loads((owner.parent/'registration-strain.json').read_bytes())['materials']
    src=(dest/'NumiHumanBedPressureTable.metalinc').read_text()
    step=float.fromhex(re.search(r'bedPressureStep=([^;]+)f;',src).group(1))
    pairs=re.findall(r'float2\(([^,]+)f,([^)]+)f\)',src)
    actual=np.array([[float.fromhex(a),float.fromhex(b)] for a,b in pairs])
    wanted=np.array([[1-compressed_stretch(i*step,m[str(j)]['registered']['source_mu_pa'],
        m[str(j)]['registered']['source_logJ_coefficient_pa']) for j in (15,16)] for i in range(257)])
    if actual.shape!=wanted.shape or np.max(abs(actual-wanted))>3e-7 or abs(actual[-1,1]-1/3)>1e-6:
        raise ValueError('Registered pressure lookup disagrees with current material')
    previous=(prior/vis).read_text()
    v=correct_order((owner/vis).read_text())
    a='        const bool useRegionalThickness ='
    v=replace(v,a,excerpt(previous,'        std::vector<mr_float4> resolvedStock;',a)+a)
    a='                std::ofstream footprintReceipt;'
    v=replace(v,a,excerpt(previous,'                std::vector<double> fullAreas;',a)+a)
    v=replace(v,'                    if (tributaryAreas[v] == 0) continue;',
              '                    if (!resolvedWitness && tributaryAreas[v] == 0 && (fullAreas.empty() || fullAreas[v]==0)) continue;')
    a='                    if(footprintReceipt.is_open()){'
    v=replace(v,a,excerpt(previous,'                    if(resolvedWitness)resolvedStock[v]',a)+a)
    a='vertex.deformationWeight.y = float(skinThicknessByRow[row] + fatThicknessByRow[row]);'
    v=replace(v,a,'vertex.deformationWeight.y = resolvedWitness ? resolvedStock[v].y+resolvedStock[v].z : float(skinThicknessByRow[row] + fatThicknessByRow[row]);')
    a='        if(bedSurface)std::cout<<"resting_bed_surface=fixed_world_triangulated"'
    v=replace(v,a,excerpt(previous,'        if(resolvedWitness) {',a)+a)
    save(vis,v)
    h=(owner/geometry).read_text();old=(prior/geometry).read_text()
    a='class NumiHumanRestingSupportGeometry final {'
    h=replace(h,a,'#include "NumiHumanResolvedBed.hpp"\n'+a+'\n    std::unique_ptr<NumiHumanResolvedBed> resolvedBed_;')
    a='        const bool firstStepDiagnostic ='
    h=replace(h,a,excerpt(old,'        if (resolvedBed_) {',a)+a)
    h=replace(h,'public:','public:\n    void installResolvedBed(std::unique_ptr<NumiHumanResolvedBed> bed) { resolvedBed_=std::move(bed); }')
    save(geometry,h)
    old=(prior/stand).read_text();h=(finite/stand).read_text()
    a='    return kUseSpeculativeContactAdmissionDistanceMeters > 0.0f'
    h=replace(h,a,excerpt(old,'    // An open unilateral column',a)+a)
    a='    if (gap >= 0.0f) {'
    h=replace(h,a,excerpt(old,'    if (support.planePoint.z == -2.0f) {',a)+a)
    save(stand,h)
    a='#include "NumiHumanRestingSupportGeometry.metalinc"'
    save('human_respiration.metal',replace((owner/'human_respiration.metal').read_text(),a,a+'\n#include "NumiHumanResolvedBed.metalinc"'))
    copy(finite,host)
    cmd=json.loads((finite/'curves.json').read_bytes())['commands'][:2]
    cmd=[[v.replace(str(finite)+'/',str(dest)+'/') for v in row] for row in cmd]
    cmd += [[v.replace(str(owner)+'/',str(dest)+'/') for v in row] for row in json.loads((owner/'build-commands.json').read_bytes())]
    if not all(str(dest) in ' '.join(row) for row in cmd):raise ValueError('Unexpected native build command')
    with (dest/'build.log').open('w')as log:
        for row in cmd:subprocess.run(row,cwd=Path(cfg['environment']['DYLD_LIBRARY_PATH']).parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    meta=dict(parent=digest(owner/'identities.json'),prior=digest(prior/'identities.json'),current_material_pressure_table_max_abs_strain_error=float(np.max(abs(actual-wanted))),
        material_pressure_ceiling_Pa=256*step,commands=cmd,inputs={p.name:digest(p) for p in dest.iterdir() if p.suffix in ('.hpp','.mm','.metalinc','.metal')},
        outputs={f:digest(dest/f) for f in ('numi-human-native','metalrobo/HumanRestingSkin.metallib','metalrobo/MetalRobo.metallib')})
    (dest/'registered-live.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('owner','finite','prior','control','dest'):p.add_argument(n,type=lambda v:Path(v).resolve())
    build(**vars(p.parse_args()))
