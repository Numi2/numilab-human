"""Generate native contact/pose data and compile the existing Metal runtime for this topology."""
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from build_cortical_skin_registration import write_contact_include

def run(work):
    q=work.parent
    src=q/'gravity-contact/app'
    v=q/'qualified-material'
    old=q.parent/'relative/volume'
    out=work/'binding'
    app=out/'app'
    rt=out/'runtime'
    for f in (app,rt,out/'contact'):f.mkdir(exist_ok=True,parents=True)
    load=lambda f:json.loads(f.read_bytes())
    points=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3))
    bed=work/'selected-pose/current.node'
    x=points(bed)
    y=points(v/'current.node')
    node=np.load(q/'seam-volume/local-ancestry.npz')['surviving_source_nodes']
    env=load(q/'gravity-scene/launch.json')['environment']
    previous_loaded=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])
    original=points(previous_loaded)
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='<u4')
    ordered=raw.copy()
    ordered[:,:4]-=1
    stamp=hashlib.sha256(ordered.tobytes()).hexdigest()
    prior=load(q.parent/'relative/binding/build.json')['ordered_nodes_and_material']
    contact=q/'registered-material-contacts/compiled'
    data=load(contact/'result.json')
    program=contact/'material-contact.npz'
    C=np.load(v/'material-walls.npz')['C']
    if data['loaded']!=digest(v/'current.node') or not np.array_equal(x[C],y[C]):raise ValueError('Moved cortical material coordinates')
    def replace(text,a,b):
        if text.count(a)!=1:raise ValueError('Unexpected native source pin '+a)
        return text.replace(a,b)
    data['loaded']=digest(bed)
    prog=out/'contact'/program.name
    if not prog.exists():prog.symlink_to(program)
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
    for key,a,b in [('nodes',len(original),len(node)),('cells',1457316,len(raw))]:
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
    tr=q/'tile-refresh';oldlib=tr/'runtime';oldapp=tr/'app'
    for folder,target,name in [(oldlib,rt,'runtime.mm'),(oldapp,app,'app-runtime.mm')]:
        text=(folder/name).read_text()
        for a,b in [(digest(old/'mesh.ele'),digest(v/'mesh.ele')),('object.elementCount!=1457316u || object.stateCount!=314015u',f'object.elementCount!={len(raw)}u || object.stateCount!={len(node)}u'),(prior[:16]+'ull',stamp[:16]+'ull'),(','.join(map(str,bytes.fromhex(prior))),','.join(map(str,bytes.fromhex(stamp))))]:
            text=replace(text,a,b)
        (target/name).write_text(text);(target/'bed_response.inc').write_bytes((folder/'bed_response.inc').read_bytes())
    cmds=load(tr/'build.json')['commands'][:3]+load(q/'gravity-contact/build.json')['commands']
    for cmd in cmds:
        for i,s in enumerate(cmd):
            cmd[i]=s.replace(str(oldlib)+'/',str(rt)+'/').replace(str(oldapp)+'/',str(app)+'/').replace(str(src)+'/',str(app)+'/')
    def compile_one(i):
        with (out/f'build-{i}.log').open('w') as f:subprocess.run(cmds[i],stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(compile_one,(0,2,3)))
    for i in [1,4]:compile_one(i)
    result=dict(commands=cmds,app=digest(app/'numi-human-native'),library=digest(rt/'libmetalrobo.dylib'),loaded=data['loaded'],elements=data['elements'],ordered=stamp,anchors=count)
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':run(Path(__file__).resolve().parent)
