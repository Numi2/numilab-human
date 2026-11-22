"""Build the existing native Human owner with an exact registered current-coordinate initializer."""
import json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from evaluate_registered_bed_transfer import digest
from build_distinct_registry import replace
from screen_registered_dense_posture import initial_body_targets

def build(src,full,donor,authoring,pose,app,out,band_path=None,physical=None):
    code=Path(__file__).resolve().parent;main='numilab_human_myosim_visual_probe.mm'
    prior=json.loads((src/'registered-cell-audit-build.json').read_text())
    if digest(src/main)!=prior['source_sha256'] or digest(src/'numi-human-native')!=prior['binary_sha256']:raise ValueError('changed native owner')
    proven=json.loads((pose/'loaded-registration.json').read_text());recovered=json.loads((pose/'recovery.json').read_text())
    band=json.loads(band_path.read_text()) if band_path else {}
    if bool(band_path)!=bool(physical):raise ValueError('An IPC pose requires its matched solver closure')
    if proven['source']!=recovered['source'] or proven['outer_rows_below_plane'] or proven['min_outer_height_m']!=band.get('minimum_m',0.) or proven['largest_attachment_error_m']>2e-7:raise ValueError('unqualified loaded bed or attachment')
    target,identity=initial_body_targets(donor)
    if any(proven['source'].get(k)!=v for k,v in identity.items()):raise ValueError('wrong skeletal operating point')
    reference=json.loads(authoring.read_text())['all_bodies']
    check=json.loads((full/'native/accepted-geometry/step-16.receipt.json').read_text())['initial_anatomical_registration']['body_poses']
    if any(np.linalg.norm(np.array(v['position_m'])-reference[v['body_index']]['current']['position_world_m'])>2e-7 or
           np.linalg.norm(np.array(v['quaternion_xyzw'])-reference[v['body_index']]['current']['orientation_world_xyzw'])>2e-7 for v in check):raise ValueError('wrong native authoring FK')
    out.mkdir(exist_ok=False)
    for f in src.iterdir():
        if f.is_file() and f.suffix in ('.hpp','.metal','.metalinc','.metallib','.inc'):(out/f.name).hardlink_to(f)
    (out/'metalrobo').symlink_to((src/'metalrobo').resolve(),target_is_directory=True)
    def vec(v):return '{'+','.join(float(x).hex()+'f' for x in v)+'}'
    def arr(rows):return '{\n'+',\n'.join(rows)+'\n}'
    refs=[f"{{{r['body_index']},{vec(r['current']['position_world_m'])},{vec(r['current']['orientation_world_xyzw'])}}}" for r in reference]
    targets=[f"{{{r['body_index']},{vec(r['position_m'])},{vec(r['quaternion_xyzw'])}}}" for r in target]
    anc=[f"{{{r['stable_attachment_id']}u,{r['volume_row']}u,{r['donor']}u,{vec(r['target_m'])}}}" for r in proven['attachment_rows']]
    header='namespace numiLoadedData {\nstruct Body{unsigned id;float p[3],q[4];};struct Anchor{unsigned stable,node,donor;float p[3];};\n'
    for k,v in [('nodeSHA',recovered['decoded_node_sha256']),('referenceSHA',proven['source']['reference_sha256']),('elementsSHA',proven['source']['elements_sha256'])]:header+=f'constexpr char {k}[]="{v}";\n'
    for k,v in [('nodes',proven['node_count']),('cells',proven['cell_count']),('outerNodes',proven['surface_rows'])]:header+=f'constexpr unsigned {k}={v};\n'
    for typ,k,v in [('Body','reference',refs),('Body','target',targets),('Anchor','anchors',anc)]:header+=f'constexpr {typ} {k}[]={arr(v)};\n'
    (out/'NumiHumanLoadedSkinPoseData.inc').write_text(header+'}\n')
    shutil.copyfile(code/'NumiHumanLoadedSkinPose.hpp',out/'NumiHumanLoadedSkinPose.hpp')
    s=(src/main).read_text()
    s=replace(s,'#include "NumiHumanRegisteredVolumeFirstTouch.hpp"','#include "NumiHumanRegisteredVolumeFirstTouch.hpp"\n#include "NumiHumanLoadedSkinPose.hpp"')
    s=replace(s,'    if(!posesAt(q,bodies)||!posesAt(model.defaultQ,restBodies))return false;','    if(!posesAt(q,bodies)||!posesAt(model.defaultQ,restBodies))return false;\n    numiLoadedSkin::sourceFrames(bodies);')
    s=replace(s,'        const auto pose=std::span<const MRBodyStateGPU>(bodies);\n        metalrobo::MetalArticulatedOperatorInput skinRestInput;',
              '        auto registrationBodies=bodies;numiLoadedSkin::sourceFrames(registrationBodies);\n        const auto pose=std::span<const MRBodyStateGPU>(registrationBodies);\n        metalrobo::MetalArticulatedOperatorInput skinRestInput;')
    # Current-world inputs in the skin source import refer to its authoring FK.
    # Their coordinates and masses are consumed before the prepared current field is installed.
    import re
    start=s.index('        const auto pose=std::span<const MRBodyStateGPU>(registrationBodies);')
    end=s.index('            std::cout<<"whole_body_skin_fat_registered_volume_install=PASS"',start)
    block=re.sub(r'(?<![.\w])bodies\b','registrationBodies',s[start:end])
    s=s[:start]+block+s[end:]
    s=replace(s,'const bool placed=checkRegisteredPhysicalOuterAtSeed(poseQ,header,physicalAlignment,alignmentError);',
              'const bool placed=numiLoadedSkin::enabled()?numiLoadedSkin::align(poseQ,header,physicalAlignment,alignmentError):checkRegisteredPhysicalOuterAtSeed(poseQ,header,physicalAlignment,alignmentError);')
    s=replace(s,'            std::cout<<"whole_body_skin_fat_registered_volume_install=PASS"','            numiLoadedSkin::install(installedObject,bundle,sidecars);\n            std::cout<<"whole_body_skin_fat_registered_volume_install=PASS"')
    s=replace(s,'!requestedRootPose.has_value() && requestedPoseCoordinates.empty()),\n                    "--resting-initial-posture requires',
              '(!requestedRootPose.has_value() || numiLoadedSkin::enabled()) && requestedPoseCoordinates.empty()),\n                    "--resting-initial-posture requires')
    s=replace(s,'                overriddenPoseQ = posture.q;\n                poseQ = overriddenPoseQ;',
              '                overriddenPoseQ = posture.q;\n                if(requestedRootPose){require(numiLoadedSkin::enabled() && overriddenPoseQ.size()>=7,"unbound loaded root placement");'
              'std::copy(requestedRootPose->begin(),requestedRootPose->end(),overriddenPoseQ.begin());}\n                poseQ = overriddenPoseQ;')
    (out/main).write_text(s)
    cmds=[[a.replace(str(src)+'/',str(out)+'/') for a in cmd] for cmd in prior['commands']]
    if band:
        from bind_barrier_initialization import bind
        bind(out,src,band,pose)
        link=cmds[-1]
        link[link.index('lib/libmetalrobo.dylib')]=str(physical/'libmetalrobo.dylib')
        original=next(v for v in link if v.endswith('/app-runtime.mm.o'))
        link[link.index(original)]=str(physical/'app-runtime.mm.o')
    for cmd in cmds:subprocess.run(cmd,check=True,cwd=app/'build-native')
    (out/'identities.json').write_text(json.dumps(dict(commands=cmds,binary_sha256=digest(out/'numi-human-native'),lib_sha256=digest(physical/'libmetalrobo.dylib') if physical else None,parent=prior['binary_sha256'],binary=digest(out/'numi-human-native'),
        generated_header=digest(out/'NumiHumanLoadedSkinPoseData.inc'),source=digest(out/main),initializer=digest(out/'NumiHumanLoadedSkinPose.hpp'),
        reference_body_dump=digest(authoring),initial_coordinate_node=recovered['decoded_node_sha256']),indent=2)+'\n')
if __name__=='__main__':build(*(Path(v).resolve() for v in sys.argv[1:]))
