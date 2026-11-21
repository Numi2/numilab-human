"""Compile the response kernels into the existing native resting shader and body transaction."""
import json,os,shutil,subprocess,sys
from pathlib import Path
from material_context import sha
src,assets,app,out=map(Path,sys.argv[1:])
out.mkdir(parents=True,exist_ok=True);(out/'metalrobo').mkdir(exist_ok=True)
for f in src.iterdir():
    if f.suffix=='.hpp' and f.name!='NumiHumanRestingCoupling.hpp' or f.suffix=='.metalinc' or f.name=='numilab_human_myosim_visual_probe.mm':
        if not (out/f.name).exists():(out/f.name).hardlink_to(f)
dest=out/'metalrobo/MetalRobo.metallib'
if not dest.exists():dest.hardlink_to(src/'metalrobo/MetalRobo.metallib')
for name in ('pelvic_contact.metalinc','pelvic_material_validate.metalinc','NumiHumanPelvicContact.hpp','pelvic_crossing_predicates.metalinc','pelvic_end_geometry.metalinc','pelvic_cortical_frame.metalinc'):
    shutil.copyfile(Path(__file__).parent/name,out/name)
manifest=json.loads((assets/'manifest.json').read_text());files=['cortical-samples','cortical-modes','cortical-triangles','cortical-neighborhood','material-nodes','material-modes','material-cells','material-inverse','pool-material-points','pool-cortical-neighborhood','pool-material-regions','cortical-free-index','cortical-flexibility','closed-material-vertices','closed-material-triangles','closed-material-shapes','closed-material-ranges','closed-material-incidence','free-node-rows','free-node-columns','free-node-stiffness']+['pelvic-cell-volume', 'pelvic-cell-incidence', 'pelvic-cell-corners', 'pelvic-factor-rows', 'pelvic-factor-columns', 'pelvic-factor-values', 'pelvic-factor-levels', 'pelvic-factor-order', 'pelvic-factor-perm', 'pelvic-factor-scale','pelvic-projector-rows','pelvic-projector-cols','pelvic-projector-vals','pelvic-reference-nodes','pelvic-reference-inverse','pelvic-reference-poses','pelvic-finite-pairs','pelvic-finite-poses']
reference_surface='surface_material_reference' in manifest
if reference_surface:
    files.append('closed-material-reference')
    count=sum(s['vertices'] for s in manifest['closed_reverse']['sides'])
    if (assets/'closed-material-reference.bin').stat().st_size!=32*count:
        raise ValueError('Skinning reference size disagrees with registered boundary')
certified='cortical_certificate' in manifest
if certified:
    if not reference_surface:raise ValueError('Cortical certificate requires the registered surface')
    files.append('cortical-certificate-faces')
    size=16*sum(r['count'] for r in manifest['cortical_certificate']['banks'])
    if size==0 or (assets/'cortical-certificate-faces.bin').stat().st_size!=size:raise ValueError('Cortical face count')
shared_facets='shared_finite_faces' in manifest
if shared_facets:
    if not certified:raise ValueError('Physical facet assembly requires its observed surface')
    files.append('shared-finite-faces')
    if (assets/'shared-finite-faces.bin').stat().st_size!=16*manifest['finite_facets']['number']:raise ValueError('Finite face map size')
forward='shared_forward_surface' in manifest
if forward:
    if not shared_facets:raise ValueError('Forward queries need the shared material/cortical registration')
    files.extend(['forward-surface-samples','forward-cortical-faces'])
    if (assets/'forward-surface-samples.bin').stat().st_size!=32*manifest['closed_reverse']['first']:raise ValueError('Forward sites')
    if (assets/'forward-cortical-faces.bin').stat().st_size*10!=(assets/'cortical-triangles.bin').stat().st_size:raise ValueError('Forward faces')
physical_boundary='constitutive_boundary' in manifest
if physical_boundary:
    if not forward:raise ValueError('Physical P1 surface requires registered pressure and cortex')
    files.append('constitutive-boundary-reference')
    if (assets/'constitutive-boundary-reference.bin').stat().st_size!=48*sum(s['vertices'] for s in manifest['closed_reverse']['sides']):
        raise ValueError('Mismatched physical P1 boundary layout')
facet_rows=manifest['finite_facets'].get('rows_per_pair',2)
assert facet_rows in (2,16)
face_spec=manifest['finite_facets']
assert face_spec['start']+facet_rows*face_spec['number']==manifest['contact_pool']['count']
assert (assets/'pelvic-finite-pairs.bin').stat().st_size==17*16*face_spec['number']
assert manifest['specimen']['mu']==12000 and manifest['specimen']['lam']==120000 and manifest['specimen']['jmin']==.20 and manifest['specimen']['jmax']==5
decl='inline constexpr bool pelvicPhysicalBoundary='+('true' if physical_boundary else 'false')+';\ninline constexpr bool pelvicHasSharedForward='+('true' if forward else 'false')+';\ninline constexpr bool pelvicHasSharedFacets='+('true' if shared_facets else 'false')+';\ninline constexpr bool pelvicHasCrossingCertificate='+('true' if certified else 'false')+';\ninline constexpr bool pelvicMaterialReferenceSurface='+('true' if reference_surface else 'false')+';\nstruct PelvicMaterialFile { const char* name; unsigned long bytes; const char* hash; };\ninline constexpr PelvicMaterialFile pelvicMaterialFiles[]={\n'
for name in files:
    p=assets/(name+'.bin');digest=sha(p);assert digest==manifest['files'][p.name]
    decl+='{"%s",%d,"%s"},\n'%(p.name,p.stat().st_size,digest)
(out/'pelvic_material_assets.hpp').write_text(decl+'};\ninline constexpr unsigned pelvicMaterialFacetRows='+str(facet_rows)+'u;\ninline constexpr unsigned pelvicUniqueVertices=%du,pelvicLeafCount=%du;\n'%(manifest['closed_reverse']['unique_vertices'],manifest['closed_reverse']['leaves']))
text=(src/'NumiHumanRestingCoupling.hpp').read_text()
def replace(what,with_):
    global text
    if text.count(what)!=1:raise ValueError(what)
    text=text.replace(what,with_)
replace('class NumiHumanRestingCoupling {','#include "NumiHumanPelvicContact.hpp"\nclass NumiHumanRestingCoupling {\n    NumiHumanPelvicContact pelvicContact;')
replace('        prepareWorld=pipeline(','''        if(std::getenv("NUMI_HUMAN_PELVIC_CONTACT_DATA")){
            auto pelvicLib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&e];
            need(pelvicLib!=nil,"pelvic material shader library missing");
            pelvicContact.configure(device,pelvicLib);
        }
        prepareWorld=pipeline(''')
replace('            auto& respiration=*physiology.respiration;\n            if(respiration.commonGeometryGateEnabled)', '            pelvicContact.encode(p,false);\n            auto& respiration=*physiology.respiration;\n            if(respiration.commonGeometryGateEnabled)')
replace('            bridge(validateBody);r.phase=EncodePhase::postCommit;', '            pelvicContact.encode(p,true);\n            bridge(validateBody);r.phase=EncodePhase::postCommit;')
replace('            auto e=[cb computeCommandEncoder];need(e!=nil,"resting accepted frame commit encoder");',
        '            pelvicContact.publish(p,numi::human::buffer(physiology.runtime.statusBuffer()),presentationCandidateRespiration);\n            auto e=[cb computeCommandEncoder];need(e!=nil,"resting accepted frame commit encoder");')
(out/'NumiHumanRestingCoupling.hpp').write_text(text)
spec=manifest['contact_flexibility']
assert sum(row['nodes'] for row in spec)*48==(assets/'material-nodes.bin').stat().st_size
surface=manifest['closed_reverse'];assert surface['leaves']<=(1<<14)
dimensions='constant uint4 pelvicSurf[2]={'+','.join('uint4(%d,%d,%d,%d)'%tuple(r[k] for k in ('first_vertex','vertices','first_face','faces')) for r in surface['sides'])+'};\nconstant uint pelvicLeaves='+str(surface['leaves'])+'u;\n'
fact=manifest['regional_factor']
def constvec(n,ty,rows):
    return 'constant '+ty+' '+n+'[2]={'+','.join(ty+'('+','.join(str(int(v))+'u' for v in row)+')' for row in rows)+'};\n'
assert len(fact)==2 and max(r['coordinates'] for r in fact)*4<=16384
dimensions+='constant uint pelvicFactorCapacity='+str(max(r['coordinates'] for r in fact))+';\n'
dimensions+=constvec('pelvicFactorDims','uint4',[[r['coordinate_start'],r['coordinates'],*r['lower']] for r in fact])
dimensions+=constvec('pelvicFactorUpper','uint2',[r['upper'] for r in fact])
dimensions+='constant uint pelvicFacetRows='+str(facet_rows)+'u;\n'
dimensions+='constant uint pelvicFiniteStart=%du;\n'%manifest['finite_facets']['start']
dimensions+='constant bool pelvicPhysicalBoundary='+('true' if physical_boundary else 'false')+';\n'
dimensions+='constant bool pelvicSharedForward='+('true' if forward else 'false')+';\n'
dimensions+='constant bool pelvicSharedFacets='+('true' if shared_facets else 'false')+';\n'
dimensions+='constant bool pelvicCachedCortex='+('true' if certified else 'false')+';\n'
dims=dimensions+'constant uint pelvicReverseStart='+str(surface['first'])+'u;\nconstant uint4 pelvicFlex[2]={'+','.join('uint4(%d,%d,%d,%d)'%tuple(r[k] for k in ('first_node','nodes','input_nodes','first_vector')) for r in spec)+'};\n'
(out/'human_respiration.metal').write_text((src/'human_respiration.metal').read_text()+dims+'\n#include "pelvic_contact.metalinc"\n#include "pelvic_material_validate.metalinc"\n#include "pelvic_end_geometry.metalinc"\n#include "pelvic_cortical_frame.metalinc"\n')
cmd=[[str(k).replace(str(src)+'/',str(out)+'/') for k in row] for row in json.loads((src/'build-commands.json').read_text())]
(out/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
for row in cmd:subprocess.run(row,cwd=app/'build-native',check=True)
(out/'identities.json').write_text(json.dumps({str(p):sha(p) for p in [src/'numi-human-native',out/'numi-human-native',out/'metalrobo/HumanRestingSkin.metallib',out/'NumiHumanRestingCoupling.hpp',out/'NumiHumanPelvicContact.hpp',out/'pelvic_material_assets.hpp',assets/'manifest.json']},indent=2)+'\n')
