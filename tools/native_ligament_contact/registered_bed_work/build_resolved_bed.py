"""Build the live per-column law and its articulated work transfer in the pinned native scene."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from export_contact_columns import change
from fit_bed_section_curve import stress
from resolve_registered_bed_load import material_stretch
from material_context import sha


def build(parent, source, curves, owner, launch, witness):
    read=lambda p:json.loads(p.read_text())
    save=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
    ids=read(parent/'identities.json')
    if any(sha(parent/k)!=v for k,v in ids.items()): raise ValueError("parent changed")
    subprocess.run(['cp','-cR',parent,owner],check=True);launch.mkdir()
    for name in ['NumiHumanResolvedBed.hpp','NumiHumanResolvedBed.metalinc']:
        shutil.copy2(Path(__file__).resolve().parent/name,owner/name)
    law=read(curves)['material']
    if any(sha(Path(k))!=v for k,v in law['material_pins'].items()):raise ValueError("source material changed")
    props=[(r['mu_pa'],r['log_volume_coefficient_pa']) for r in law['materials']]
    cap=min(float(stress(2/3,*p)) for p in props)
    levels=np.linspace(0,cap,257)
    strains=np.array([1-material_stretch(levels,m) for m in props]).T.astype('f4')
    lit=lambda v:float(v).hex()+'f'
    table='constant float bedPressureStep='+lit(cap/256)+';\nconstant float2 bedPressureTable[257]={\n'
    table+=',\n'.join('float2('+lit(a)+','+lit(b)+')' for a,b in strains)+'\n};\n'
    (owner/'NumiHumanBedPressureTable.metalinc').write_text(table)
    def edit(name,a,b):
        p=owner/name;p.write_text(change(p.read_text(),a,b))
    edit('NumiHumanBedFootprint.hpp',
        'w[0]+=twice*(3-u-v)/6.; w[1]+=twice*u/6.; w[2]+=twice*v/6.;',
        '''// A clipped fan centroid is a convex combination in the source
        // triangle. Preserve its area partition on floating-point boundaries.
        const double a0=3.-u-v;
        if (std::min({a0,u,v}) < -64.*std::numeric_limits<double>::epsilon())
            throw std::domain_error("clipped footprint centroid outside source face");
        const std::array<double,3> bary={std::max(0.,a0),std::max(0.,u),std::max(0.,v)};
        const double scale=0.5*twice/(bary[0]+bary[1]+bary[2]);
        for (unsigned k=0;k<3;++k) w[k]+=scale*bary[k];''')
    name='NumiHumanRestingSupportGeometry.hpp'
    edit(name,'class NumiHumanRestingSupportGeometry final {','#include "NumiHumanResolvedBed.hpp"\nclass NumiHumanRestingSupportGeometry final {\n    std::unique_ptr<NumiHumanResolvedBed> resolvedBed_;')
    edit(name,'public:','public:\n    void installResolvedBed(std::unique_ptr<NumiHumanResolvedBed> bed) { resolvedBed_=std::move(bed); }')
    edit(name,'        const bool firstStepDiagnostic =','''        if (resolvedBed_) {
            if (environmentCount_!=1 || bed_.counts.z || predictedEndpoint ||
                pass.groundPoint.z!=0 || pass.groundNormal.z!=1 ||
                pass.groundNormal.x!=0 || pass.groundNormal.y!=0)
                return reject("unsupported resolved bed geometry phase");
            resolvedBed_->encode(command,worldZKeys_,contacts,pointWorld,pointLow,
                                 pointJacobian,dispatch,pass.stepIndex);
        }
        const bool firstStepDiagnostic =''')
    name='NumiHumanRestingVisual.hpp'
    edit(name,'        const bool useRegionalThickness =','        std::vector<mr_float4> resolvedStock;\n        const char* resolvedWitness=std::getenv("NUMI_HUMAN_RESOLVED_BED_WITNESS");\n        const bool useRegionalThickness =')
    edit(name,'                if (const char* output=std::getenv("NUMI_HUMAN_COMPLETE_BED_FOOTPRINT_CSV")) {','''                const char* output=std::getenv("NUMI_HUMAN_COMPLETE_BED_FOOTPRINT_CSV");
                if(resolvedWitness)resolvedStock.resize(posedSkin.size());
                if (output || resolvedWitness) {''')
    edit(name,'                    complete.open(output);','                    if(output){complete.open(output);')
    edit(name,'<<std::setprecision(17);\n                }\n                std::ofstream footprintReceipt;','<<std::setprecision(17);}\n                }\n                std::ofstream footprintReceipt;')
    edit(name,'if (tributaryAreas[v] == 0 && (fullAreas.empty() || fullAreas[v]==0)) continue;', 'if (!resolvedWitness && tributaryAreas[v] == 0 && (fullAreas.empty() || fullAreas[v]==0)) continue;')
    edit(name,'                    if (!fullAreas.empty() && fullAreas[v]>0) {','''                    if(resolvedWitness)resolvedStock[v]={float(fullAreas[v]),float(kWholeBodySkinThicknessM[nearest]),float(kWholeBodyFatThicknessM[nearest]),0};
                    if (output && fullAreas[v]>0) {''')
    edit(name,'vertex.deformationWeight.y = float(skinThicknessByRow[row] + fatThicknessByRow[row]);','vertex.deformationWeight.y = resolvedWitness ? resolvedStock[v].y+resolvedStock[v].z : float(skinThicknessByRow[row] + fatThicknessByRow[row]);')
    edit(name,'        if(bedSurface)std::cout<<"resting_bed_surface=fixed_world_triangulated"','''        if(resolvedWitness) {
            require(!bedSurface && !resolvedStock.empty() && !organSupportActive && !ropeConfig &&
                    supportMapSpan.size()==skin.header.vertexCount && support.header.groundPointZ==0,"registered bed source mismatch");
            const auto path=humanBrainMetallibPath().parent_path()/"HumanRestingSkin.metallib";
            auto materialLibrary=[device newLibraryWithURL:[NSURL fileURLWithPath:@(path.c_str())] error:&e];
            require(materialLibrary!=nil,"cannot load registered bed GPU kernels");
            skinSupport->installResolvedBed(std::make_unique<NumiHumanResolvedBed>(device,materialLibrary,resolvedStock,
                supportMapSpan,supportWeightSpan,regionForVertex,unsigned(regions.size()),
                supportInfluenceLayout==NumiHumanSkinInfluenceLayout::tile32?32u:1u,resolvedWitness));
            std::cout<<"resting_bed_force=individual_live_vertex_gap skin_fat_equal_traction=1 pressure_weighted_articulated_jacobian=1"<<std::endl;
        }
        if(bedSurface)std::cout<<"resting_bed_surface=fixed_world_triangulated"''')
    name='NumiHumanStand.metal'
    edit(name,'    return kUseSpeculativeContactAdmissionDistanceMeters > 0.0f','''    // An open unilateral column carries no traction in the current linearization.
    if (support.planePoint.z == -2.0f && support.planePoint.w > 0.0f && support.planePoint.x == 0.0f)
        return -INFINITY;
    return kUseSpeculativeContactAdmissionDistanceMeters > 0.0f''')
    edit(name,'    if (gap >= 0.0f) {','''    if (support.planePoint.z == -2.0f) {
        const float force=support.planePoint.x,tangent=support.planePoint.y;
        if (!(force>0.0f && tangent>0.0f && isfinite(force) && isfinite(tangent))) return {NAN,NAN};
        return {support.boundaryNormalVelocity + force/(tangent*interval),
                1.0f/(tangent*dt*interval)};
    }
    if (gap >= 0.0f) {''')
    name='human_respiration.metal'
    edit(name,'#include "NumiHumanRestingSupportGeometry.metalinc"','#include "NumiHumanRestingSupportGeometry.metalinc"\n#include "NumiHumanResolvedBed.metalinc"')
    cfg=read(source/'launch.json')
    cfg['environment'].pop('NUMI_HUMAN_COMPLETE_BED_FOOTPRINT_CSV',None)
    cfg['environment']['NUMI_HUMAN_RESOLVED_BED_WITNESS']=str(witness)
    for group in ('build-commands.json','shader-build-commands.json'):
        old=read(parent/group)
        marker=old[0].index('-c')
        shader_dir=Path(old[0][marker+1]).parent if Path(old[0][marker+1]).suffix=='.metal' else parent
        programs=[[v.replace(str(shader_dir)+'/',str(owner)+'/') for v in cmd] for cmd in old]
        declared=owner/('numi-human-native' if group=='build-commands.json' else 'metalrobo/MetalRobo.metallib')
        if programs[-1][-1]!=str(declared) and str(declared) not in programs[-1]:raise ValueError('build destinations differ')
        save(owner/group,programs)
        for cmd in programs:subprocess.run(cmd,cwd=Path(cfg['environment']['DYLD_LIBRARY_PATH']).parent,check=True)
    names=set(ids)|{'NumiHumanResolvedBed.hpp','NumiHumanResolvedBed.metalinc','NumiHumanBedPressureTable.metalinc'}
    save(owner/'identities.json',{k:sha(owner/k) for k in sorted(names)})
    cfg['parent_launch']=sha(source/'launch.json');cfg['pins'][str(curves)]=sha(curves);save(launch/'launch.json',cfg)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','source','curves','owner','launch','witness'):p.add_argument(key,type=lambda s:Path(s).resolve())
    build(**vars(p.parse_args()))
