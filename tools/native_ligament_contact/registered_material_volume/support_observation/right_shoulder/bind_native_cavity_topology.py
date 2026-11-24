"""Compile the measured topology contract for an imported volume into each active native owner."""
import hashlib,json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from admit_native_cavity import read,write,require,replace_once,digest,tets

def topology(p):
    a=tets(p).astype('<u4');a[:,:4]-=1
    return hashlib.sha256(a.tobytes()).digest(),len(a),int(a[:,:4].max()+1)

def build(w):
    reg=w/'native-binding';base=w.parent/'conditioned/witness-material/volume';v=w/'material-import/volume'
    retained=reg/'before-direction.json'
    if not retained.exists():retained.write_bytes((reg/'build.json').read_bytes())
    spec=read(retained);commands=spec['commands']
    before=topology(base/'mesh.ele');after=topology(v/'mesh.ele')
    require(before[1:]==(spec['original_cells'],spec['original_nodes']) and after[2]==spec['new_source_node'],'Wrong nodal domain')
    pairs={};generated={};cmds=[]
    for link,objname in [(commands[1],'runtime.mm.o'),(commands[3],'app-runtime.mm.o')]:
        obj=next(Path(s)for s in link if Path(s).name==objname);src=obj.with_suffix('');z=read(obj.parent.parent/'build.json')
        c=next(a for a in z['commands']if str(src) in a and str(obj)in a and '-c'in a);s=src.read_text()
        old=digest(base/'mesh.ele');new=digest(v/'mesh.ele')
        s=replace_once(s,'std::string_view(geometry)=="'+old+'"','std::string_view(geometry)=="'+new+'"')
        s=replace_once(s,f'object.elementCount!={before[1]}u || object.stateCount!={before[2]}u',f'object.elementCount!={after[1]}u || object.stateCount!={after[2]}u')
        literal=lambda b:'const unsigned char expected[]={'+','.join(map(str,b))+'};'
        s=replace_once(s,literal(before[0]),literal(after[0]))
        out=reg/src.name;out.write_text(s);pairs[str(obj)]=str(reg/obj.name);generated[src.name]=[digest(src),digest(out)]
        args=[str(out)if x==str(src)else pairs[str(obj)]if x==str(obj)else x for x in c];args[1:1]=['-iquote',str(src.parent)];cmds.append(args)
    def compile(k,c):
        with (reg/f'compile-direction-{k}.log').open('w')as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2)as pool:
        jobs=[pool.submit(compile,k,c)for k,c in enumerate(cmds)]
        for job in jobs:job.result()
    for k,link in enumerate([commands[1],commands[3]],2):
        c=[pairs.get(a,a)for a in link];compile(k,c);cmds.append(c)
    spec.update(binary=digest(reg/'numi-human-native'),library=digest(reg/'libmetalrobo.dylib'),direction=dict(input=before[0].hex(),compiled=after[0].hex(),cells=after[1],nodes=after[2],source=generated,commands=cmds))
    write(reg/'build.json',spec);print('Authenticated local connectivity/material rows',after[1],after[0].hex(),flush=True)
if __name__=='__main__':build(Path.cwd())
