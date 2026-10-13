#!/usr/bin/env python3
"""Prepare existing native resting owner with a reviewed passive skin registration."""
from pathlib import Path
import argparse, hashlib, json, shutil, sys, os
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v: raise ValueError(m)
def save(p,j): p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
def main():
    ap=argparse.ArgumentParser()
    for key in ('parent','candidate','audit','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args(); parent=a.parent.resolve(); out=a.output.resolve()
    require(not out.exists(),'preserve prior output')
    audit=json.loads(a.audit.read_text()); gates=audit['pose_gates']
    require(gates['all8_skin_self_exact_zero'] and gates['all8_skin_vs_92_targets_exact_zero'] and gates['capture_count']==8,'candidate gates failed')
    require(audit['changes']['sha256']==sha(a.candidate),'candidate changed')
    require(audit['changes']['normals_bindings_weights_topology_byte_identical'] and audit['changes']['ocular_115_vertices_bit_identical'],'undeclared change')
    decl=json.loads((parent/'baseline/run-declaration.json').read_text())
    pins=dict(decl['immutable_assets'])
    for p,h in pins.items(): require(sha(p)==h,'source drift '+p)
    src=decl['argv']; mi=src.index('numilab_human.resting_run')
    tail=src[mi+1:].copy(); oldscene=Path(tail[tail.index('--body-scene')+1])
    oldreceipt=Path(tail[tail.index('--anatomy-receipt')+1])
    scene=json.loads(oldscene.read_text()); receipt=json.loads(oldreceipt.read_text())
    oldskin=scene['source']['skin']['path']; oldsha=scene['source']['skin']['sha256']
    require(oldsha==audit['pins']['source_skin'],'different baseline skin')
    out.mkdir(); (out/'baseline').mkdir(); (out/'assets').mkdir()
    for old in (parent/'assets').iterdir():
        if old.is_file(): os.link(old,out/'assets'/old.name)
    for key in ('map','polynomials','domain_boxes'):
        rec=receipt['provenance']['cardiac_geometry_binding']['common_field'][key]
        rel=Path(rec['path']); require(not rel.is_absolute() and '..' not in rel.parts,'invalid cardiac path')
        require(sha(parent/rel)==rec['sha256'],'cardiac source drift')
        dst=out/rel; dst.parent.mkdir(parents=True,exist_ok=True); os.link(parent/rel,dst)
    skin=out/'assets/bodyparts3d-myosim-skinned-shell.nhskin'; shutil.copy2(a.candidate,skin)
    # All existing physical support/contact and pose declarations are retained.
    scene['source']['skin'].update(path=str(skin),sha256=sha(skin))
    correction={'candidate_path':str(a.candidate.resolve()),'candidate_sha256':sha(skin),
        'audit_path':str(a.audit.resolve()),'audit_sha256':sha(a.audit),
        'mixed_source_reference':True,'not_measured_individual':True,
        'source_changes':audit['changes'],'native_qualification':'pending',
        'scope':'Localized inferred passive medial-thigh skin registration; no force, pose, mass or controller changes.'}
    scene['medial_thigh_reference_correction']=correction
    sp=out/'resting-supine-scene.manifest.json'; save(sp,scene)
    require(receipt['mass_geometry_accounting']['skin_payload_sha256']==oldsha,'receipt baseline differs')
    receipt['mass_geometry_accounting']['skin_payload_sha256']=sha(skin)
    receipt['mass_geometry_accounting']['medial_thigh_reference_correction']=correction
    receipt['qualification']='Reviewed skin registration on retained back412 anatomy; native verification pending; unresolved left forearm remains excluded from anatomical admission.'
    rp=out/'resting-anatomy-receipt.json'; save(rp,receipt)
    tail[tail.index('--body-scene')+1]=str(sp)
    tail[tail.index('--anatomy-receipt')+1]=str(rp)
    tail[tail.index('--output')+1]=str(out/'baseline/native-run')
    env=dict(x.split('=',1) for x in src[2:mi-2])
    env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(out/'baseline/native-run/common-field-failure.json')
    sys.path.insert(0,env['PYTHONPATH'])
    from numilab_human import resting_run as rr
    parser=argparse.ArgumentParser(); rr.add_arguments(parser); ns=parser.parse_args(tail)
    native,assets=rr.command(ns)
    oldnative=decl['owner_cli_preview']['native_argv']
    require(len(native)==len(oldnative),'native cardinality changed')
    changes=[{'old':x,'new':y} for x,y in zip(oldnative,native) if x!=y]
    oldout=Path(src[src.index('--output')+1]).resolve()
    allowed={str(oldscene.resolve()),str(oldreceipt.resolve()),str(Path(oldskin).resolve()),str(oldout),str(oldout/'native-viewer.mov')}
    require(all(x in allowed for x,y in zip(oldnative,native) if x!=y),'unexpected physical argument change')
    launch=out/'launch_arm.py'; shutil.copy2(parent/'launch_arm.py',launch)
    retention=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/input-retention-1380/receipt.json')
    compiled=json.loads(retention.read_text())['compiled_snapshot']
    for r in compiled: pins[r['retained_path']]=r['sha256']
    for p in (Path(__file__),a.candidate,a.audit,sp,rp,launch,retention): pins[str(p.resolve())]=sha(p)
    pins.update(assets)
    argv=['/usr/bin/env','-i',*[k+'='+v for k,v in sorted(env.items())],src[mi-2],'-m','numilab_human.resting_run',*tail]
    decl.update(argv=argv,immutable_assets=pins,qualification=receipt['qualification'],
        launch_adapter={'path':str(launch),'sha256':sha(launch)},
        owner_cli_preview={'native_argv':native,'asset_sha256':assets,'recorded_invocation_environment':rr.invocation_environment(env)})
    dp=out/'baseline/run-declaration.json'; save(dp,decl)
    save(out/'launch-guard.json',{'launcher_sha256':sha(launch),'arms':{'baseline':{'run_declaration_sha256':sha(dp),'native_output':str(out/'baseline/native-run')}}})
    save(out/'preparation.json',{'parent':str(parent),'skin_sha256':sha(skin),'native_argv_changes':changes,'qualification':receipt['qualification'],'compiled_source_retention_path':str(retention)})
    require(all(sha(p)==h for p,h in pins.items()),'input changed')
    print(json.dumps({'output':str(out),'pins':len(pins),'native_argv_changes':changes}))
if __name__=='__main__': main()
