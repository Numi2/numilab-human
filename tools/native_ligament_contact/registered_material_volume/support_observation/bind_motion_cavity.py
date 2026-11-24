"""Rebind registered surface ancestry and the native topology certificate for a fixed-boundary cavity."""
import csv,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from build_bed_audit import replace_once,sha
from build_cortical_skin_registration import write_contact_include

def run(parent,case,solver,observer,stage,out):
    out.mkdir(exist_ok=True);v=stage/'volume';old=parent/'volume';ccfg=json.loads((case/'launch.json').read_bytes());env=ccfg['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    for name in ('current.node','mesh.node','material-walls.npz'):
        if sha(old/name)!=sha(v/name):raise ValueError('Material surface coordinates or topology changed')
    basis=json.loads((stage/'material-p1/result.json').read_bytes())
    if any(rr['inadmissible_label_rows'] for rr in basis['roles'].values()) or json.loads((stage/'native-replay/result.json').read_bytes())['exit']:raise ValueError('Unqualified mass allocation')
    o=np.load(v/'overlay.npz');mapping=o['old_to_new'];removed=np.unique(o['source']);added=np.unique(o['target'])
    read=lambda p:np.loadtxt(p/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    before=read(old);after=read(v);before[:,:4]-=1;after[:,:4]-=1
    def ordered(c):return hashlib.sha256(c.astype('<u4',copy=False).tobytes()).hexdigest()
    origsha,updatedsha=map(ordered,(before,after));oldsha,newsha=sha(old/'mesh.ele'),sha(v/'mesh.ele')
    if np.any(before[mapping>=0]!=after[mapping[mapping>=0]]):raise ValueError('Surviving topology changed')
    terms={}
    def repair_csv():
        for key in ('COMBINED_NODE_MAP','FACET_ANCESTRY'):
            orig=Path(env[pre+key+'_CSV']);dest=out/(key.lower()+'.csv')
            if sha(orig)!=env[pre+key+'_SHA256']:raise ValueError('Unmatched ancestry')
            with orig.open()as f,dest.open('w')as g:
                reader=csv.reader(f);writer=csv.writer(g,lineterminator='\n');writer.writerow(next(reader))
                for a in reader:
                    at=43 if key=='COMBINED_NODE_MAP' else 18;c=int(a[at]);new=int(mapping[c])
                    if key=='COMBINED_NODE_MAP':
                        if new<0:
                            n=int(a[0])-1;options=[j for j in added if n in after[j,:4] and after[j,4]==before[c,4]]
                            if not options or any(float(z) for z in a[48:52]):raise ValueError('Unrepresentable operational ancestry')
                            new=int(min(options));corner=np.flatnonzero(after[new,:4]==n).item();a[44:48]=[str(int(i==corner))for i in range(4)]
                    else:
                        if int(a[17])!=c+1:raise ValueError('Mismatched facet parent')
                        if new<0:
                            face=set(int(x)-1 for x in a[19:22]);options=[j for j in added if face.issubset(after[j,:4]) and after[j,4]==before[c,4]]
                            if len(options)!=1:raise ValueError('Ambiguous anatomical surface')
                            new=int(options[0])
                        a[17]=str(new+1)
                        if 'headwall' in a[3] and a[16]=='[-1,-1,-1]':
                            if json.loads(a[23])!=[c]:raise ValueError('Unexpected headwall ancestry')
                            a[23]=json.dumps([new])
                    a[at]=str(new);writer.writerow(a)
            terms[pre+key+'_CSV']=str(dest);terms[pre+key+'_SHA256']=sha(dest)
    def build():
        app=out/'app';rt=out/'runtime';con=out/'contact'
        for d in (app,rt,con):d.mkdir(exist_ok=True)
        original=parent/'binding/app'
        data='NumiHumanLoadedSkinPoseData.inc';text=(original/data).read_text();text=replace_once(text,oldsha,newsha);text=replace_once(text,'unsigned cells='+str(len(before))+';','unsigned cells='+str(len(after))+';');(app/data).write_text(text)
        # The parent loaded-pose header includes its data by relative path.
        shutil.copyfile(original/'NumiHumanLoadedSkinPose.hpp',app/'NumiHumanLoadedSkinPose.hpp')
        src=parent/'binding/contact';c=json.loads((src/'result.json').read_bytes());z=dict(np.load(src/'material-contact.npz'));faces=z['facets'];old_rows=faces[:,0].copy()
        if np.any(mapping[old_rows]<0):raise ValueError('Cavity intersects an explicit cortical contact certificate')
        faces[:,0]=mapping[old_rows];program=con/'material-contact.npz';np.savez_compressed(program,**z);c.update(elements=newsha,program=sha(program));(con/'result.json').write_text(json.dumps(c,indent=2)+'\n');write_contact_include(con,app)
        b=json.loads((solver/'build.json').read_bytes());obs=json.loads((observer/'build.json').read_bytes());rc=[]
        for where,target,name in ((solver/'runtime',rt,'runtime.mm'),(solver/'app',app,'app-runtime.mm')):
            text=(where/name).read_text()
            for a,d in [(oldsha,newsha),(f'object.elementCount!={len(before)}u',f'object.elementCount!={len(after)}u'),(origsha[:16]+'ull',updatedsha[:16]+'ull'),(','.join(map(str,bytes.fromhex(origsha))),','.join(map(str,bytes.fromhex(updatedsha))))]:
                text=replace_once(text,a,d)
            (target/name).write_text(text);shutil.copyfile(where/'bed_response.inc',target/'bed_response.inc')
        for i in (0,1,2):
            cmd=[x.replace(str(solver/'runtime')+'/',str(rt)+'/').replace(str(solver/'app')+'/',str(app)+'/')for x in b['commands'][i]]
            if i in (0,2):cmd[1:1]=['-iquote',str(solver/('runtime' if i==0 else 'app'))]
            rc.append(cmd)
        for name in ('numilab_human_myosim_visual_probe.mm','NumiHumanRestingVisual.hpp'):shutil.copyfile(observer/'app'/name,app/name)
        for original_cmd in obs['commands'][2:]:
            rc.append([x.replace(str(observer/'app')+'/',str(app)+'/').replace(str(solver/'runtime')+'/',str(rt)+'/').replace(str(solver/'app')+'/',str(app)+'/')for x in original_cmd])
        def compile(i):
            with (out/f'build-{i}.log').open('w') as f:subprocess.run(rc[i],stdout=f,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(3) as pool:list(pool.map(compile,[0,2,3]))
        compile(1);compile(4)
        target=rt/'libmetalrobo.dylib';exe=app/'numi-human-native'
        if str(target)not in rc[4] or str(exe)not in rc[4]:raise ValueError('Wrong physical link')
        return dict(commands=rc,library=sha(target),executable=sha(exe),updated_cells=len(after),cortical_facets=len(faces),prior=oldsha,topology=newsha,ordered_hash=updatedsha)
    with ThreadPoolExecutor(2)as pool:
        fut=pool.submit(repair_csv);b=build();fut.result()
    b['tables']=terms;(out/'build.json').write_text(json.dumps(b,indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
