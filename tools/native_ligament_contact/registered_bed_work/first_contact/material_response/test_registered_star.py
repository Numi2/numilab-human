"""Audit the installed material pose and exercise it with the admitted physical scene."""
import ctypes,json,os,sys,subprocess
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from measure_bone_tissue_overlap import run as cortical_check
from layer_contacts import run as layer_check
from certify_transported_material import measure
from run_cavity_motion import execute_native

def run(job,h,out,fit,layerlib):
    rd=lambda p:json.loads(p.read_bytes());vol=out/'relative/volume';control=out/'relative/scene';build=out/'binding'
    st=rd(fit/'result.json');proof=rd(job/'verified-connected-material/interval-audit.json')
    read=lambda f:np.ascontiguousarray(np.loadtxt(f,skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8'))
    x,y0,y=read(vol/'mesh.node'),read(vol/'current.node'),read(fit/'current.node')
    t=np.ascontiguousarray(np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1)
    protected=np.load(build/'cortex/registered-cortical-facets.npz')['cell_side_material_bank_nodes'][:,4:]
    if not np.array_equal(y[protected],y0[protected]) or st['source']!=digest(vol/'current.node') or st['elements']!=digest(vol/'mesh.ele') or st['current']!=digest(fit/'current.node'):raise ValueError('Cortical geometry, reference or topology changed')
    parent=Path((job/'storage/interval-checker-locator.txt').read_text().split()[1])
    if digest(parent)!=proof['checker'] or digest(layerlib)!=proof['layer_checker']:raise ValueError('Unknown predicate')
    fn=ctypes.CDLL(str(parent)).scanCellIntervals;fp=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    fn.argtypes=[fp,fp,ctypes.c_size_t,ip,ctypes.c_size_t,fp];v=np.empty((len(t),3))
    if fn(x,y,len(x),t,len(t),v):raise ValueError('Native interval failure')
    margin=np.minimum(v[:,0]-v[:,1]-.3500000,2.5-v[:,0]-v[:,1]);bad=np.flatnonzero((~np.isfinite(margin))|(margin<=0))
    if len(bad):raise ValueError('Uncertified intervals '+str(bad[:20]))
    z=np.load(job/'registered-boundary/registered-boundary.npz')
    if any(not np.array_equal(y[z[k+'_node_joint']],y0[z[k+'_node_joint']]) for k in 'OD'):raise ValueError('Outer or deep surface changed')
    layer_check(y,[z[k+'_node_joint'][z[k+'_face_nodes']] for k in 'OID'],fit/'layers.json',layerlib)
    if np.any(rd(fit/'layers.json')['pair_counts']):raise ValueError('Starting layer intersection')
    changed=np.flatnonzero(np.any(x!=y,axis=1));field=fit/'actual-transport.npz';np.savez_compressed(field,nodes=changed,positions=y[changed].astype('f4'))
    cortical_check(h,fit/'bone-overlap',job/'distal-registered-native/target-pose.json',field,vol)
    old=rd(job/'verified-connected-material/bone-overlap/result.json')['results'];new=rd(fit/'bone-overlap/result.json')['results']
    if set(new)!=set(old) or len(new)!=185:raise ValueError('Bone identity changed')
    increase=[(k,m,new[k][m]-old[k][m]) for k in old for m in ('skin','fat') if new[k][m]>old[k][m]+1e-6]
    (fit/'quality.json').write_text(json.dumps(dict(cells=len(t),minimum_interval_margin=float(margin.min()),failed_intervals=len(bad),audited_bones=len(new),increased_bone_overlap_mm3=increase),indent=2)+'\n')
    if increase:raise ValueError('Increased local cortical overlap: '+str(increase))
    app=fit/'app';app.mkdir();controls=build/'app';main='numilab_human_myosim_visual_probe.mm'
    for p in controls.iterdir():
        if p.suffix not in ('.h','.hpp','.inc','.mm'):continue
        if p.name in ['NumiHumanLoadedSkinPoseData.inc','NumiHumanCorticalSkinBinding.inc']:
            s=p.read_text()
            if s.count(st['source'])!=1:raise ValueError('Wrong loaded geometry pin')
            (app/p.name).write_text(s.replace(st['source'],st['current']))
        elif p.name==main:
            s=p.read_text();anchor='transaction.restingFEMRegionLabelFirstTetrahedron=\n                            sharedWorld.objects[owner.wholeBodySkinObjectIndex].elementOffset;'
            if s.count(anchor)!=1:raise ValueError('Shared physical span owner missing')
            s=s.replace(anchor,anchor+'''
                        const auto& materialSpan=sharedWorld.objects[owner.wholeBodySkinObjectIndex];
                        std::cerr<<"registered_material_span object="<<owner.wholeBodySkinObjectIndex
                            <<" element_offset="<<materialSpan.elementOffset
                            <<" element_count="<<materialSpan.elementCount
                            <<" state_offset="<<materialSpan.stateOffset
                            <<" state_count="<<materialSpan.stateCount<<"\\n";''')
            (app/p.name).write_text(s)
        else:os.link(p,app/p.name)
    commands=[[w.replace(str(controls)+'/',str(app)+'/') for w in c] for c in rd(build/'build.json')['commands'][-2:]]
    with (fit/'build.log').open('w') as f:
        for c in commands:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
    scene=fit/'scene';scene.mkdir();cfg=rd(control/'launch.json');env=cfg['environment'];cfg['parent_launch']=digest(control/'launch.json')
    for k,w in list(env.items()):
        if w.startswith(str(control)+'/') and w not in cfg['pins']:env[k]=str(scene/Path(w).relative_to(control))
    env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(fit/'current.node')
    cfg['argv'][0]=str(app/'numi-human-native');cfg['argv'][4]=str(scene/'native-traces')
    for f in [fit/'current.node',app/'numi-human-native']:cfg['pins'][str(f)]=digest(f)
    for w in set(cfg['argv']+list(env.values()))&set(cfg['pins']):
        if digest(Path(w))!=cfg['pins'][w]:raise ValueError('Changed input '+w)
    (fit/'build.json').write_text(json.dumps(dict(commands=commands,binary=digest(app/'numi-human-native'),material=st),indent=2)+'\n')
    (scene/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(cfg['argv'],cfg['cwd'],clean|env,scene,minimum_free_bytes=4*2**30,timeout_s=600),flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
