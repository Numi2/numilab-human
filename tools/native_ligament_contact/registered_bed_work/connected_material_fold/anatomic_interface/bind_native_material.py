"""Generate native contact/pose data and compile the existing Metal runtime for this topology."""
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from build_cortical_skin_registration import write_contact_include

def run(work,refined=False):
    q=work.parent
    stage=work/'witness-material' if refined else work
    src=work/'binding/app' if refined else q/'gravity-contact/app'
    v=stage/'volume' if refined else q/'qualified-material'
    old=q/'qualified-material' if refined else q.parent/'relative/volume'
    out=stage/'binding'
    app=out/'app'
    rt=out/'runtime'
    for f in (app,rt,out/'contact'):f.mkdir(exist_ok=True,parents=True)
    load=lambda f:json.loads(f.read_bytes())
    points=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    bed=v/'current.node' if refined else work/'selected-pose/current.node'
    x=points(bed)
    y=points(work/'selected-pose/current.node' if refined else v/'current.node')
    node=np.arange(len(x)-1) if refined else np.load(q/'seam-volume/local-ancestry.npz')['surviving_source_nodes']
    env=load(work/'native-integrated-01/launch.json' if refined else q/'gravity-scene/launch.json')['environment']
    previous_loaded=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])
    original=points(previous_loaded)
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='<u4')
    ordered=raw.copy()
    ordered[:,:4]-=1
    stamp=hashlib.sha256(ordered.tobytes()).hexdigest()
    prior=load(work/'binding/build.json')['ordered'] if refined else load(q.parent/'relative/binding/build.json')['ordered_nodes_and_material']
    contact=work/'binding/contact' if refined else q/'registered-material-contacts/compiled'
    data=load(contact/'result.json')
    program=contact/'material-contact.npz'
    C=np.load(v/'material-walls.npz')['C']
    if data['loaded']!=digest(work/'selected-pose/current.node' if refined else v/'current.node') or not np.array_equal(x[C],y[C]):raise ValueError('Moved cortical material coordinates')
    def replace(text,a,b):
        if text.count(a)!=1:raise ValueError('Unexpected native source pin '+a)
        return text.replace(a,b)
    data['loaded']=digest(bed)
    prog=out/'contact'/program.name
    if refined:
        sample=dict(np.load(program))
        if np.isin(sample['facets'][:,0],np.load(v/'source-overlay.npz')['old_cells']).any():raise ValueError('Remeshed cavity contains an explicit cortical facet')
        data.update(material_reference=digest(v/'mesh.node'),elements=digest(v/'mesh.ele'))
    if not prog.exists():prog.symlink_to(program.resolve())
    (out/'contact/result.json').write_text(json.dumps(data,indent=2)+'\n')
    write_contact_include(out/'contact',app)
    inc='NumiHumanLoadedSkinPoseData.inc'
    for f in src.iterdir():
        if f.suffix in ('.h','.hpp','.mm','.inc') and f.name not in (inc,'NumiHumanCorticalSkinBinding.inc'):
            # Keep direct sources beside the generated includes; quoted includes resolve locally.
            dst=app/f.name
            if not dst.exists():dst.write_bytes(f.read_bytes())
    text=(src/inc).read_text()
    for k,path,new in [('nodeSHA','current.node',bed),('referenceSHA','mesh.node',v/'mesh.node'),('elementsSHA','mesh.ele',v/'mesh.ele')]:
        text=replace(text,k+'[]="'+digest(previous_loaded if path=='current.node' else old/path)+'"',k+'[]="'+digest(new)+'"')
    for key,a,b in [('nodes',len(original),len(x)),('cells',1457308 if refined else 1457316,len(raw))]:
        text=replace(text,'unsigned '+key+'='+str(a)+';','unsigned '+key+'='+str(b)+';')
    ri=np.full(len(original),-1,'i4');ri[node]=np.arange(len(node))
    def index(i):
        n=ri[i]
        if n<0 or not np.array_equal(x[n],original[i]):raise ValueError('Moved registered attachment or bed node '+str(i))
        return n
    def anchor(m):
        stable,n,donor=m.groups()
        return '{'+stable+'u,'+str(index(int(n)))+'u,'+donor+'u,{'
    text,count=re.subn(r'^\{(\d+)u,(\d+)u,(\d+)u,\{',anchor,text,flags=re.M)
    if not count:raise ValueError('No physical anchors')
    for i in [24331,24538,26199]:
        if index(i)!=i:raise ValueError('Bed registration changed')
    (app/inc).write_text(text)
    tr=work/'binding' if refined else q/'tile-refresh';oldlib=tr/'runtime';oldapp=tr/'app'
    old_counts=f'object.elementCount!={1457308 if refined else 1457316}u || object.stateCount!={len(original)}u'
    for folder,target,name in [(oldlib,rt,'runtime.mm'),(oldapp,app,'app-runtime.mm')]:
        text=(folder/name).read_text()
        for a,b in [(digest(old/'mesh.ele'),digest(v/'mesh.ele')),(old_counts,f'object.elementCount!={len(raw)}u || object.stateCount!={len(x)}u'),(prior[:16]+'ull',stamp[:16]+'ull'),(','.join(map(str,bytes.fromhex(prior))),','.join(map(str,bytes.fromhex(stamp))))]:
            text=replace(text,a,b)
        (target/name).write_text(text);(target/'bed_response.inc').write_bytes((folder/'bed_response.inc').read_bytes())
    cmds=load(tr/'build.json')['commands'] if refined else load(tr/'build.json')['commands'][:3]+load(q/'gravity-contact/build.json')['commands']
    for cmd in cmds:
        for i,s in enumerate(cmd):
            cmd[i]=s.replace(str(oldlib)+'/',str(rt)+'/').replace(str(oldapp)+'/',str(app)+'/').replace(str(src)+'/',str(app)+'/')
    def compile_one(i):
        with (out/f'build-{i}.log').open('w') as f:subprocess.run(cmds[i],stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(compile_one,(0,2,3)))
    for i in [1,4]:compile_one(i)
    result=dict(commands=cmds,app=digest(app/'numi-human-native'),library=digest(rt/'libmetalrobo.dylib'),loaded=data['loaded'],elements=data['elements'],ordered=stamp,anchors=count)
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':run(Path(__file__).resolve().parent,'--motion-witness' in sys.argv[1:])
