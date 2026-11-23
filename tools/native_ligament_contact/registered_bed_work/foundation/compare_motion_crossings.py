"""Attribute the same declared skin-anatomy triangle pair across opening and accepted motion."""
import hashlib,json,runpy,subprocess,sys
from pathlib import Path
from verify_native_extensor_pair import loaded
from material_context import sha
def compare(case,reader,bank,opening):
    sys.path.insert(0,str(reader.parent/'src'));api=runpy.run_path(str(reader));scan=runpy.run_path(str(bank))['scan_banks']
    out=case/'opening-envelope';out.mkdir(exist_ok=True)
    rec=json.loads((case/'terminal.receipt.json').read_text()); full=json.loads((opening/'terminal.receipt.json').read_text())
    if rec['base_pack_file_sha256']!=full['base_pack_file_sha256']:raise ValueError('different opening geometry')
    def q(path):
        line,=[l for l in (path/'stdout.log').read_text().splitlines() if l.startswith('stand_terminal_state=')]
        return json.loads(line.split('=',1)[1])['initial_q']
    if q(case)!=q(opening):raise ValueError('initial q differs')
    poses={p['body_index']:p for p in full['initial_anatomical_registration']['body_poses']}
    if sorted(poses)!=list(range(rec['accepted_body_pose_count'])):raise ValueError('incomplete reference skeleton')
    for p in rec['initial_anatomical_registration']['body_poses']:
        if p!=poses[p['body_index']]:raise ValueError('different common opening body pose')
    def original():
        raw=subprocess.check_output(['zstd','-qdc',case/'initial.mrvpack.zst'])
        if hashlib.sha256(raw).hexdigest()!=rec['base_pack_file_sha256']:raise ValueError('changed opening frame')
        return raw
    raw=original()
    ((_,_,skin,faces),)=api['primitives'](raw,poses,skin_only=True)
    result=dict(pack_sha256=rec['base_pack_file_sha256'],step=0,registered_pose_count=len(poses),skin_vertices=len(skin),skin_faces=len(faces))
    scan(api,raw,poses,skin,faces,result,out,'opening.json',check_skin=True)
    del raw,skin,faces
    subprocess.run([sys.executable,bank.parent/'classify_registered_envelope.py',out/'opening.json',case/'stdout.log',out/'physical-banks.json','--window-receipt',case/'resting-ocular-sclera-window.json'],check=True)
    adv=case/'envelope';rows=[]
    for target,name in [(out,'opening.json'),(adv,'accepted-envelope.json')]:
        j=json.loads((target/'physical-banks.json').read_text());r=json.loads((target/name).read_text())
        w={}
        for l in (target/'target-witnesses.jsonl').read_text().splitlines():
            f=json.loads(l)
            if f['ocular_source_id'] or (f['semantic']==51010 and f['stable_id']>=4294966700):continue
            k=(f['semantic'],f['stable_id'],f['primitive'])
            if k in w:raise ValueError('duplicate bank')
            w[k]=set(map(tuple,f['triangle_pairs']))
        if sum(map(len,w.values()))!=j['physical_nonocular_pairs']:raise ValueError('pair counts changed')
        rows.append((j,r,w))
    (a,ra,wa),(b,rb,wb)=rows
    # Pair identity requires identical source triangle topology for these two captures.
    def topology(initial):
        states,raw=(poses,original()) if initial else loaded(case,False,False)[1:]
        return {(sem,stable):hashlib.sha256(faces.tobytes()).hexdigest() for sem,stable,xyz,faces in api['primitives'](raw,states)}
    if topology(True)!=topology(False):raise ValueError('changed anatomical triangle topology')
    stats=[]
    for k in sorted(wa.keys()|wb.keys()):
        x,y=wa.get(k,set()),wb.get(k,set())
        stats.append(dict(semantic=k[0],stable_id=k[1],primitive=k[2],opening=len(x),accepted=len(y),persistent=len(x&y),introduced=len(y-x),cleared=len(x-y)))
    total={k:sum(z[k] for z in stats) for k in ('opening','accepted','persistent','introduced','cleared')}
    data=dict(accepted_step=rec['accepted_step'],accepted_s=rec['accepted_time_s'],opening_self_intersections=ra['skin_self_intersections'],accepted_self_intersections=rb['skin_self_intersections'],
        opening=sha(out/'physical-banks.json'),accepted=sha(adv/'physical-banks.json'),pairs=total,
        most_introduced=sorted(stats,key=lambda r:-r['introduced'])[:12],scope='Exact intersecting triangles at two endpoints with identical connectivity. Pairs can change as a persistent contact moves across a face. This is not a swept clearance proof.')
    (case/'crossing-attribution.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':compare(*(Path(x).resolve() for x in sys.argv[1:]))
