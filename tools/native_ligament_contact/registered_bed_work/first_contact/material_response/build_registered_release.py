"""Recompile the existing first-contact owner for a registered volume at the matched release pose."""
import json,os,re,subprocess,sys
from pathlib import Path
import numpy as np
from register_material_surfaces import sha,tetgen_positions,pinned


def run(control,previous,build,mesh,volume,mass,out):
    cfg=json.loads(control.read_bytes());env=cfg['environment'];src=previous/'owner'
    ident=json.loads((src/'identities.json').read_bytes());base=json.loads((build/'build.json').read_bytes())
    oldref=pinned(env,'TETGEN_NODE','TETGEN_NODE_SHA256');oldcur=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])
    if sha(oldref)!=sha(oldcur):raise ValueError('A deformed release needs explicit displacement and history transfer')
    imported=json.loads((volume/'native-acceptance.json').read_bytes())
    result=json.loads((mass/'result.json').read_bytes())
    if result['exit'] or len(result['receipts'])!=2:raise ValueError('Mass certification missing')
    current=mesh/'mesh.node';elements=mesh/'mesh.ele'
    target=imported['settings']
    pref='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    if (sha(current)!=target[pref+'TETGEN_NODE_SHA256'] or sha(elements)!=target[pref+'TETGEN_ELEMENTS_SHA256']
            or sha(mesh/'current-child.labels.csv')!=target[pref+'CHILD_LABELS_SHA256']):
        raise ValueError('Wrong registered volume')
    y=tetgen_positions(current);x=tetgen_positions(oldref)
    rec=json.loads((previous/'installation.json').read_bytes())
    if not np.array_equal(y,y.astype('f4').astype('f8')) or rec['mode']!='first contact under normal gravity' or rec['material_translation_m']!=0.:
        raise ValueError('Release initialization is different')
    out.mkdir()
    dataset='NumiHumanLoadedSkinPoseData.inc'
    for p in src.iterdir():
        if p.suffix in {'.hpp','.h','.mm','.metalinc','.inc','.metal'} and p.name!=dataset:
            os.link(p,out/p.name)
    old_header=(src/dataset).read_text()
    def swap(txt,search,new):
        s,n=re.subn(search,new,txt)
        if n!=1:raise ValueError('Ambiguous release registration')
        return s
    header=old_header
    for key,f,old in [('nodeSHA',current,oldcur),('referenceSHA',current,oldref),
                       ('elementsSHA',elements,pinned(env,'TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256'))]:
        header=swap(header,'constexpr char '+key+r'\[\]="'+sha(old)+'";','constexpr char '+key+'[]="'+sha(f)+'";')
    for name,before,after in [('nodes',len(x),len(y)),('cells',imported['control']['volume_cells'],imported['rebuilt']['volume_cells'])]:
        header=swap(header,'constexpr unsigned '+name+'='+str(before)+';',f'constexpr unsigned {name}={after};')
    # Every physical contact candidate and deep attachment keeps the original point.
    bed=rec['bed_z_m'];width=float(np.float32(1e-5))
    from csv import DictReader
    combined=Path(target[pref+'COMBINED_NODE_MAP_CSV'])
    if sha(combined)!=target[pref+'COMBINED_NODE_MAP_SHA256']:raise ValueError('Changed boundary')
    with combined.open() as stream:
        outer=np.asarray([int(r['joint_node_id'])-1 for r in DictReader(stream) if r['role']=='O'])
    contacts=outer[(y[outer,2]-bed)<width]
    oldcontacts=sorted(map(int,re.search(r'registeredBedContactNodes\[\]=\{([^}]+)',header).group(1).replace('u','').split(',')))
    anchors=list(map(int,re.findall(r'\{\d+u,(\d+)u,\d+u,\{',header)))
    if (len(outer)!=int(re.search(r'constexpr unsigned outerNodes=(\d+);',header).group(1)) or
            min(y[:,2])-bed!=rec['initial_gap_m'] or sorted(contacts.tolist())!=oldcontacts or len(anchors)!=14 or
            not np.array_equal(x[anchors],y[anchors]) or not np.array_equal(y[contacts],x[contacts])):
        raise ValueError('Registered first contact or attachment coordinates differ')
    (out/dataset).write_text(header)
    app=out/'numi-human-native.o';prog=out/'numi-human-native'
    old_obj=src/app.name
    if sha(build/'numi-human-native')!=base['binary'] or sha(build/'libmetalrobo.dylib')!=base['library']:raise ValueError('Changed native runtime')
    compile=list(ident['commands'][0]);compile[compile.index('-c')+1]=str(out/Path(compile[compile.index('-c')+1]).name)
    compile[compile.index('-o')+1]=str(app)
    link=list(base['commands'][-1])
    if str(old_obj) not in link:raise ValueError('Actual entry object differs')
    link[link.index(str(old_obj))]=str(app);link[link.index('-o')+1]=str(prog)
    for c in (compile,link):subprocess.run(c,check=True)
    info=lambda f:dict(path=str(f),sha256=sha(f))
    result=dict(commands=[compile,link],executable=info(prog),release_pose=info(current),library=info(build/'libmetalrobo.dylib'),
                generated_header=info(out/dataset),source_header=info(src/dataset),upstream_launch=sha(control),
                original_release_equals_reference=True,release_mode=rec['mode'],release_height_m=bed,
                first_contact_node_ids=contacts.tolist(),initial_gap_m=rec['initial_gap_m'],identical_attachment_positions=len(anchors),
                previous_release_nodes=len(x),current_nodes=len(y),material_elements=imported['rebuilt']['volume_cells'],
                native_mass_result=sha(mass/'result.json'))
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
