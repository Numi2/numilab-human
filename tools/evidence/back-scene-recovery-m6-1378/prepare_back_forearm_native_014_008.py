#!/usr/bin/env python3
"""Prepare a bounded back-interface diagnostic with verified right177 anatomy."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
PARENT=A/'forearm-shared-field-native-012-diagonal097'
NATIVE='147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174'
def require(v,m):
    if not v:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--composition',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=16.0);ap.add_argument('--capture-steps',default='0,3807,4767,5023,5599,6207,7423,8000');args=ap.parse_args()
    require(16.0<=args.seconds<=60.0 and abs(args.seconds/.002-round(args.seconds/.002))<1e-8,'duration outside bounded integral-step diagnostic')
    steps=int(round(args.seconds/.002))
    captures=[int(x) for x in args.capture_steps.split(',')]
    require(0<len(captures)<=8 and len(set(captures))==len(captures) and all(0<=x<=steps and (x in (0,steps) or x%32==31) for x in captures),'invalid bounded captures or native capture cadence')
    comp=args.composition.resolve();out=args.output.resolve()
    require(not out.exists(),'preserve prior output')
    prep=json.loads((comp/'preparation.json').read_text())
    require('loader_preflight_passed' in prep['status'],'composition did not pass loader preflight')
    require(all(x['max_um']<=2 and x['native_float32_gate_failures']==0 for x in prep['rest_rows'].values()),'rest gate not passed')
    decl=json.loads((PARENT/'baseline/run-declaration.json').read_text())
    require(decl['capture_steps']==[0,3807,4767,5023,5599,6207,7423,8000] and decl['accepted_steps']==8000,'parent pose grid changed')
    pins={}
    def pin(p,digest=None):
        p=Path(p).resolve();value=sha(p)
        require(digest is None or value==digest,'input hash changed '+str(p))
        require(str(p) not in pins or pins[str(p)]==value,'conflicting pin')
        pins[str(p)]=value
    for d in (decl['immutable_assets'],prep['input_sha256']):
        for p,h in d.items():pin(p,h)
    pin(comp/'preparation.json');pin(comp/'rest-preflight.json');pin(Path(__file__))
    rightbundle=A/'right177-topology-bundle-001'
    rightprep=json.loads((rightbundle/'preparation.json').read_text())
    pin(rightbundle/'preparation.json')
    bound=Path('/Users/n/numi-human-fds132-m6-1350/attempt-178-right-pentagon-exact-deviation-bound-001/result-003/report.json')
    pin(bound,'814614912b2f11654e44a96cf72f7e1de131365b26353c7e152b8ed693d7475d')

    for p,h in rightprep['inputs_sha256'].items():pin(p,h)
    pin(rightprep['derived_face_lineage_path'],rightprep['derived_face_lineage_sha256'])
    pin(A/'bilateral-forearm-composition-preflight-013-006/preparation.json')

    pin(PARENT/'baseline/run-declaration.json');pin(PARENT/'preparation.json')
    require(sha(comp/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')==prep['child_payload_sha256'],'child changed')
    require(sha(comp/'passive-tissue-parent-bindings.json')==prep['extension_sha256'],'extension changed')
    out.mkdir();(out/'baseline').mkdir();(out/'assets').mkdir()
    def link(p,d):
        require(p.is_file() and not d.exists(),'bad link input/output')
        d.parent.mkdir(parents=True,exist_ok=True);os.link(p,d);pin(p);pin(d,sha(p))
    for p in (comp/'assets').iterdir():link(p,out/'assets'/p.name)
    link(comp/'passive-tissue-parent-bindings.json',out/'passive-tissue-parent-bindings.json')
    receipt=json.loads((PARENT/'resting-anatomy-receipt.json').read_text())
    for key in ('map','polynomials','domain_boxes'):
        rec=receipt['provenance']['cardiac_geometry_binding']['common_field'][key]
        rel=Path(rec['path']);require(not rel.is_absolute() and '..' not in rel.parts,'invalid common cardiac path')
        require(sha(PARENT/rel)==rec['sha256'],'cardiac source changed')
        link(PARENT/rel,out/rel)
    child=out/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue';manifest=child.with_suffix('.manifest.json')
    sys.path.insert(0,'/Users/n/numi-human-left-pq-publication-1344/src')
    from numilab_human import passive_attachment_composition as pc
    pin(Path(pc.__file__))
    ar=out/'append-receipt'/'resting-anatomy-receipt.json'
    ar.parent.mkdir()
    cr=out/'composition-receipt'/'resting-anatomy-receipt.json'
    cr.parent.mkdir()
    right_receipt=out/'right-receipt'/'resting-anatomy-receipt.json'
    right_receipt.parent.mkdir()
    pc.bind_anatomy_receipt(PARENT/'resting-anatomy-receipt.json',A/'bilateral-forearm-composition-preflight-013-006/assets'/child.name,right_receipt)
    pin(right_receipt)
    pc.bind_anatomy_receipt(right_receipt,comp/'append'/child.name,ar)
    receipt=pc.bind_anatomy_receipt(ar,child,cr)
    pin(ar);pin(cr)
    extension=json.loads((out/'passive-tissue-parent-bindings.json').read_text())
    require(extension['child_payload_sha256']==sha(child),'parent extension child mismatch')
    receipt['passive_tissue_parent_bindings']=extension
    receipt['back_native_diagnostic_014']={'status':'prepared_experimental_not_admitted','composition_path':str(comp),'composition_sha256':sha(comp/'preparation.json'),'known_cpu_limitations':'Experimental shared back interface. Source gates do not establish native deformation validity; every declared capture requires degeneracy, topology and interface audit.','scope':'Six back muscles are appended through existing composition and full source-parent GPU binding. The geometry is diagnostic. No physical force, mass, physiology or controller changes. CPU predictor residuals are retained; actual GPU geometry must be checked.','anatomical_admission_claimed':False}
    receipt['right_forearm_conditioning_177']={'candidate_path':rightprep['candidate_path'],'candidate_sha256':rightprep['candidate_sha256'],'lineage_path':rightprep['derived_face_lineage_path'],'lineage_sha256':rightprep['derived_face_lineage_sha256'],'interface_extension':rightprep['interface_extension'],'known_failure':rightprep['known_failure'],'anatomical_admission_claimed':False}
    receipt['right_forearm_source_patch_bound']={'report_path':str(bound),'sha256':sha(bound),'scope':'Exact source-space symmetric Hausdorff bound, not deformed-pose error.'}
    receipt['qualification']='Bounded native014 whole-body back-interface diagnostic from '+comp.name+' with unchanged right-forearm177. Right177 passed the separate retained one-minute run; this new back composition remains unadmitted. No full anatomical admission or long-run qualification.'

    rp=out/'resting-anatomy-receipt.json';rp.write_text(json.dumps(receipt,separators=(',',':'))+'\n');pin(rp)
    require(rp.stat().st_size<=128*1024*1024,'receipt exceeds native limit')
    src=list(decl['argv']);mi=src.index('numilab_human.resting_run')
    tail=src[mi+1:]
    require(tail[tail.index('--dt')+1]=='0.002','unexpected physical dt')
    tail[tail.index('--seconds')+1]=str(args.seconds)
    tail[tail.index('--anatomy-receipt')+1]=str(rp)
    tail[tail.index('--output')+1]=str(out/'baseline/native-run')
    env=dict(x.split('=',1) for x in src[2:mi-2])
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=','.join(map(str,captures))
    env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(out/'baseline/native-run/common-field-failure.json')
    sys.path.insert(0,env['PYTHONPATH'])
    from numilab_human import resting_run as rr
    parser=argparse.ArgumentParser();rr.add_arguments(parser);ns=parser.parse_args(tail)
    native,assets=rr.command(ns)
    require(sha(native[0])==NATIVE,'unexpected native binary')
    for p,h in assets.items():pin(p,h)
    argv=['/usr/bin/env','-i',*[k+'='+v for k,v in sorted(env.items())],src[mi-2],'-m','numilab_human.resting_run',*tail]
    # Require physical invocation invariant apart from exact anatomy asset/output paths.
    oldnative=decl['owner_cli_preview']['native_argv']
    require(len(oldnative)==len(native),'native argument cardinality changed')
    changes=[(x,y) for x,y in zip(oldnative,native) if x!=y]
    require(all(str(PARENT.resolve()) in x or str(PARENT) in x or 'forearm-shared-field-native-012-diagonal097' in x or (i>0 and oldnative[i-1]=='--muscle-step-count' and x=='8000' and y==str(steps)) for i,(x,y) in enumerate(zip(oldnative,native)) if x!=y),'unexpected nonanatomy invocation change')
    launcher=out/'launch_arm.py';shutil.copy2(PARENT/'launch_arm.py',launcher);pin(launcher)
    newdecl={'schema':decl['schema'],'seconds':args.seconds,'accepted_steps':steps,'capture_steps':captures,'execution_host':'numi','hardware':decl['hardware'],'qualification':receipt['qualification'],'immutable_assets':pins,'argv':argv,'launch_adapter':{'path':str(launcher),'sha256':sha(launcher)},'owner_cli_preview':{'native_argv':native,'asset_sha256':assets,'recorded_invocation_environment':rr.invocation_environment(env)}}
    dp=out/'baseline/run-declaration.json';save(dp,newdecl)
    guard={'launcher_sha256':sha(launcher),'arms':{'baseline':{'run_declaration_sha256':sha(dp),'native_output':str(out/'baseline/native-run')}}}
    save(out/'launch-guard.json',guard)
    command=f'/Users/n/numi-human-prep-venv-20261005/bin/python3.13 {launcher} baseline'
    (out/'launch-command.txt').write_text(command+'\n')
    save(out/'preparation.json',{'status':'prepared_experimental_not_admitted','parent_scene_path':str(PARENT),'composition_path':str(comp),'composition_sha256':sha(comp/'preparation.json'),'child_payload_sha256':sha(child),'extension_sha256':sha(out/'passive-tissue-parent-bindings.json'),'run_declaration_sha256':sha(dp),'immutable_input_count':len(pins),'input_sha256':pins,'native_argv_changes':changes,'launch_command':command,'seconds':args.seconds,'accepted_steps':steps,'gpu_run_performed':False,'anatomical_admission_claimed':False})
    require(all(sha(p)==h for p,h in pins.items()),'input changed during preparation')
    print(json.dumps({'status':'prepared_experimental_not_admitted','output':str(out),'launch_command':command,'pins':len(pins),'receipt_bytes':rp.stat().st_size}))
if __name__=='__main__':main()
