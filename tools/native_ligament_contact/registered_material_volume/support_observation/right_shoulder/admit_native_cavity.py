"""Bind a certified interior replacement to its native donor, contact and material owners."""
import csv,json,os,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from bind_interior_scene import closest
from run_cavity_motion import execute_native
from build_gluteal_interface import replace_once

def read(p):return json.loads(p.read_bytes())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def xyz(p):return np.loadtxt(p,skiprows=1,usecols=(1,2,3))
def tets(p):return np.loadtxt(p,skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
def require(v,reason):
    if not v:raise ValueError(reason)

def prepare(w):
    j=w.parent;case=j/'posterior-interface-082/native-pair';cfg=read(case/'launch.json');env=cfg['environment']
    pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    reg=w/'native-binding';reg.mkdir(exist_ok=True)
    vol=w/'material-import/volume';base=j/'conditioned/witness-material/volume'
    r=read(w/'material-import/result.json');src=read(w/'skin-fat-cavities/result.json')['edits'];car=w/'material-ancestry/carrier-replay'
    require(len(r['admitted'])==len(src)==1 and r['exit']==0 and read(car/'native-replay/result.json')['exit']==0,'Missing cavity or mass proof')
    for k,v in r['pins'].items():require(digest(vol/k)==v,k+' changed after material import')
    source=r['admitted'][0];ids=source['source'];dest=source['destination_rows']
    e=src[0];i=e['cells'][0][0];pids=e['parent_nodes'];weights=e['barycentric']
    current=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']);oldref=base/'mesh.node';oldtet=base/'mesh.ele'
    X=xyz(vol/'mesh.node');Y=xyz(vol/'current.node')
    for f,A in ((oldref,X),(current,Y)):
        B=xyz(f);require(len(A)==len(B)+1 and i==len(B) and np.array_equal(B,A[:-1]),'Moved existing registered nodes')
        require(np.max(abs(np.asarray(weights)@B[pids]-A[i]))<5e-8,'Unmatched new material point')
    old,new=tets(oldtet),tets(vol/'mesh.ele')
    require(np.array_equal(np.flatnonzero(np.any(old!=new[:len(old)],axis=1)),np.sort(ids)) and len(new)==len(old)+len(dest)-len(ids),'Changed other tetrahedra')
    require(digest(base/'material-walls.npz')==digest(vol/'material-walls.npz'),'Changed tissue boundary')
    app=j/'precision-initial';host=j/'posterior-interface-082/native-build'
    data=(app/'NumiHumanCorticalSkinBinding.inc').read_text();facets=[int(a)for a in re.findall(r'^\{(\d+),\d+,\d+,\d+,\d+,\{',data,re.M)]
    require(len(facets)>0 and not set(facets).intersection(ids),'Changed registered cortical interface')
    lab=np.loadtxt(vol/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    require(np.all(lab[dest,1:]==lab[dest[0],1:]) and len(lab)==len(new),'Crossed material or donor')
    def csvfile(k):
        f=Path(env[pre+k+'_CSV']);require(digest(f)==env[pre+k+'_SHA256'],'Stale '+k);return f
    P=np.loadtxt(env['NUMI_H706_FRACTIONAL_SOURCE330_NODES'],skiprows=1,delimiter=',',usecols=tuple(range(2,8)))
    T=np.loadtxt(env['NUMI_H706_FRACTIONAL_SOURCE330_TETS'],skiprows=1,delimiter=',',usecols=(3,4,5,6),dtype='i4')
    ancestry=np.loadtxt(csvfile('PARENT_LABELS'),skiprows=1,delimiter=',',dtype='i8')
    mapping=csvfile('COMBINED_NODE_MAP');outmap=reg/'combined_node_map.csv';found={};maxsrc=0;rebound=[]
    def bind(row,node):
        available=[k for k in dest if node+1 in new[k,:4] and int(lab[k,2])==int(row[40]) and int(lab[k,3])==int(row[41])]
        require(available,'Missing incident typed cell');cell=min(available);corner=list(new[cell,:4]).index(node+1)
        row[43:52]=[str(cell)]+[str(int(k==corner))for k in range(4)]+['0']*4
    with mapping.open(newline='')as f,outmap.open('w',newline='')as g:
        g.write(f.readline())
        for line in f:
            rec=line.rstrip('\r\n').split(',');node=int(rec[0])-1
            if rec[1]=='fat':maxsrc=max(maxsrc,int(rec[2]))
            if node in pids:found[node]=rec[:]
            if int(rec[43]) in ids:bind(rec,node);g.write(','.join(rec)+'\n');rebound.append(node)
            else:g.write(line)
        require(set(found)==set(pids) and len({tuple(v[k]for k in (12,13,40,41))for v in found.values()})==1,'Conflicting parent donors')
        rr=[v for v in found.values()if v[8]=='interior'];require(len(rr)==1,'Ambiguous interior semantics')
        row=rr[0][:];row[0]=str(i+1);row[2]=str(maxsrc+1)
        require(row[18]=='none' and row[42]=='-1','Surface provenance on appended interior')
        p1=car/'material-p1';geo=np.load(p1/'fat-role-geometry.npz')
        idx=np.flatnonzero(geo['combined_node_rows']==i);require(len(idx)==1,'Missing appended P1 point')
        nz=np.load(read(p1/'result.json')['environment']['NUMI_H706_FRACTIONAL_P1_FAT_NPZ'])
        u=int(idx[0]);require(np.array_equal(nz['points_world_m_f32'][u],X[i].astype('f4')),'Different P1 coordinate')
        a,b=nz['node_group_offsets'][u:u+2];offset=nz['node_group_parent_tet_offsets'];candidates=set()
        for k in range(a,b):
            if np.array_equal(nz['group_tuples'][nz['node_group_ids'][k]],lab[dest[0],1:4]):
                candidates.update(nz['node_group_parent_tet_rows'][offset[k]:offset[k+1]].tolist())
        opts=[(closest(X[i],P[T[k],3:]),int(k))for k in sorted(candidates) if np.array_equal(ancestry[k,1:4],lab[dest[0],1:4])]
        require(opts,'Missing source-parent group')
        (dist,bary),parent=min(opts,key=lambda a:(a[0][0],a[1]));delta=X[i]-bary@P[T[parent],:3]
        require(np.linalg.norm(delta)<=float(env[pre+'MAX_PARENT_REGISTRATION_DISTANCE_M']),'Source projection exceeds bound')
        with csvfile('DONOR_FRAMES').open()as f:ds={int(d['donor']):d for d in csv.DictReader(f)}
        d=ds[int(row[12])];R=np.array([[float(d[f'current_R{k}{l}'])for l in range(3)]for k in range(3)]);o=np.array([float(d['current_'+v])for v in 'xyz']);local=np.linalg.solve(R,X[i]-o)
        require(np.linalg.norm(R@local+o-X[i])<1e-12,'Donor registration')
        fmt=lambda a:[format(x,'.17g')for x in a]
        row[9:12]=fmt(X[i]);row[14:17]=fmt(local);row[29:39]=[str(parent)]+fmt(bary)+fmt(delta)+fmt([np.linalg.norm(delta)])+[str(sum(abs(x[0][0]-dist)<=1e-12 for x in opts)-1)]
        bind(row,i);g.write(','.join(row)+'\n')
    remaps={digest(old):digest(vol/name)for old,name in [(oldref,'mesh.node'),(oldtet,'mesh.ele'),(current,'current.node')]}
    for p in app.iterdir():
        if p.suffix not in ('.mm','.inc','.hpp','.h'):continue
        if p.name in ('NumiHumanLoadedSkinPoseData.inc','NumiHumanCorticalSkinBinding.inc'):
            s=p.read_text()
            for before,after in remaps.items():s=replace_once(s,before,after)
            if p.name=='NumiHumanLoadedSkinPoseData.inc':
                for k,b,a in [('nodes',len(X)-1,len(X)),('cells',len(old),len(new))]:s=replace_once(s,f'{k}={b};',f'{k}={a};')
            (reg/p.name).write_text(s)
        else:shutil.copy2(p,reg/p.name)
    require(str(host)==env['DYLD_LIBRARY_PATH'].split(':')[0] and digest(host/'NumiMatter.metallib')==cfg['pins'][str(host/'NumiMatter.metallib')],'Wrong GPU owner')
    hs=read(host/'build.json');s=(host/'numi_human.mm').read_text()
    require(hs['operator']==env['NUMI_HUMAN_GLUTEAL_OPERATOR'],'Changed force port')
    (reg/'numi_human.mm').write_text(replace_once(s,'candidate->nodeLoads.size()!=321219','candidate->nodeLoads.size()!=321220'))
    cmds=[]
    for c in hs['commands'][2:]:
        cmds.append([str(reg/Path(a).name) if a in {str(host/k)for k in ('numi_human.mm','numi_human.mm.o','libmetalrobo.dylib')}else a for a in c])
    cmds += [[a.replace(str(app)+'/',str(reg)+'/')for a in c]for c in read(app/'build.json')['commands']]
    for k,c in enumerate(cmds):
        with (reg/f'compile-{k}.log').open('w')as f:subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT)
    changed={pre+'COMBINED_NODE_MAP_CSV':str(outmap),pre+'COMBINED_NODE_MAP_SHA256':digest(outmap),'NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE':str(vol/'current.node')}
    for k,name in [('TETGEN_NODE','mesh.node'),('TETGEN_ELEMENTS','mesh.ele'),('CHILD_LABELS','current-child.labels.csv')]:
        changed[pre+k]=str(vol/name);changed[pre+k+'_SHA256']=digest(vol/name)
    for f in ('material-p1/result.json','packets/result.json','native-replay/result.json'):changed.update(read(car/f)['environment'])
    write(reg/'build.json',dict(parent=digest(case/'launch.json'),commands=cmds,original_nodes=len(X)-1,original_cells=len(old),protected_cortical_facets=len(facets),rebound=rebound,new_source_node=i+1,source_tet=parent,source_distance_m=np.linalg.norm(delta),donor=int(row[12]),source_partition_id=maxsrc+1,source_rows=pids,source_weights=weights,operational_cell=int(row[43]),carrier=digest(car/'native-replay/result.json'),environment=changed,binary=digest(reg/'numi-human-native'),library=digest(reg/'libmetalrobo.dylib')))
    print('Bound and compiled',i+1,parent,row[12],len(rebound),flush=True)

def launch(w,out):
    case=w.parent/'posterior-interface-082/native-pair';reg=w/'native-binding';spec=read(reg/'build.json');cfg=read(case/'launch.json');pins=cfg['pins'];env=cfg['environment'];args=cfg['argv']
    require(spec['parent']==digest(case/'launch.json') and env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1','Changed control')
    changed=spec['environment'];paths={};hashes={}
    for k,s in changed.items():
        orig=env.get(k)
        if orig in pins and s.startswith('/') and Path(s).is_file():
            require(digest(Path(orig))==pins[orig],'Changed original input '+k);h=digest(Path(s));before=pins[orig]
            require(before not in hashes or hashes[before]==h,'Ambiguous remap '+k);hashes[before]=h;paths[orig]=s;pins[s]=h
    for k,s in list(env.items()):env[k]=paths.get(s,hashes.get(s,s))
    env.update(changed);env['DYLD_LIBRARY_PATH']=str(reg)+':'+env['DYLD_LIBRARY_PATH']
    for k,s in list(env.items()):
        if s.startswith(str(case)+'/') and s not in pins:env[k]=str(out/Path(s).relative_to(case))
    args[0]=str(reg/'numi-human-native');args[4]=str(out/'native-traces')
    pins[args[0]]=spec['binary'];pins[str(reg/'libmetalrobo.dylib')]=spec['library']
    for v in set(args+list(env.values())+[str(reg/'libmetalrobo.dylib')])&set(pins):require(digest(Path(v))==pins[v],'Stale native input '+Path(v).name)
    cfg['material_cavity']=digest(reg/'build.json');out.mkdir();write(out/'launch.json',cfg)
    clean={k:v for k,v in os.environ.items()if not k.startswith(('NUMI_','NM_'))}
    result=execute_native(args,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=1000);write(out/'execution.json',result);print(result,flush=True)
if __name__=='__main__':
    w=Path.cwd()
    if len(sys.argv)==1:prepare(w)
    else:launch(w,(w/sys.argv[1]).resolve())
