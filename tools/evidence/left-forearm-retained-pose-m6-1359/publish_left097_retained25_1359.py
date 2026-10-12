#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil,platform,sys
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
SRC=A/'left097-retained25-cpu-audit-001'
PUB=Path('/Users/n/numi-human-left-pq-publication-1344')
OUT=PUB/'tools/evidence/left-forearm-retained-pose-m6-1359'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v:raise RuntimeError(m)
def save(p,j):p.write_text(json.dumps(j,sort_keys=True,indent=2)+'\n')
def main():
    report=json.loads((SRC/'report.json').read_text())
    require(len(report['results'])==25,'incomplete pose grid')
    require(not OUT.exists(),'preserve previous publication')
    for p,d in report['input_sha256'].items():require(sha(p)==d,'source changed '+p)
    sourceaudit=Path('/Users/n/numi-human-fds132-m6-1350/attempt-099-candidate097-full-source-and-late-audit-001/report.json')
    expected=json.loads(sourceaudit.read_text())['source_full_audits']['candidate097']
    signatures={k:[(x['face_count'],x['vertex_count'],x['sign']) for x in v['topology']['component_signed_volumes']] for k,v in expected.items()}
    closures=[]
    for row in report['results']:
        for k,v in row['meshes'].items():
            got=[(x['face_count'],x['vertex_count'],x['sign']) for x in v['topology']['component_signed_volumes']]
            closures.append({'arm':row['arm'],'step':row['step'],'mesh':k,'component_structure_and_volume_signs_match_source':got==signatures[k]})
    status='passed' if report['status']=='passed' and all(c['component_structure_and_volume_signs_match_source'] for c in closures) else 'failed'
    OUT.mkdir(parents=True)
    files=[SRC/'report.json',SRC/'inputs.json',A/'audit_left097_retained25_cpu_001.py',Path(__file__)]
    for row in report['results']:files.append(SRC/(row['arm']+'-'+str(row['step'])+'.json'))
    copied=[]
    for p in files:
        dst=OUT/p.name;shutil.copy2(p,dst)
        require(sha(dst)==sha(p),'copy mismatch')
        copied.append({'file':dst.name,'sha256':sha(dst),'source':str(p)})
    external={**report['input_sha256'],str(sourceaudit):sha(sourceaudit)}
    save(OUT/'component-volume-sign-audit.json',{'status':'passed' if all(c['component_structure_and_volume_signs_match_source'] for c in closures) else 'failed','source_audit_sha256':sha(sourceaudit),'source_signatures':signatures,'checks':closures,'scope':'Component identity and signed-volume orientation; this does not establish physical mass, density or material-volume conservation.'})
    save(OUT/'publication.json',{'status':status,'execution_host':'numi','scope':'CPU accepted-pose geometry audit of left097. Not new GPU execution or final anatomy admission.','copied_files':copied,'external_pinned_inputs':external,'python':sys.version,'platform':platform.platform(),'target_rows':[132,134],'pose_count':25,'mesh_audit_count':75,'elapsed_seconds':report['elapsed_seconds'],'native_evidence':'../native-anatomy-left-forearm-diagonal-m6-1358/README.md'})
    text=f"""# Left forearm across 25 retained poses

Result: **{status}**. The source-preserving diagonal repair 097 was deformed through all 25 retained body-pose snapshots on SSH numi. Each pose checks both left forearm muscles and their joined external surface: 75 complete exact Float32-lattice checks for self-intersections, degenerate faces, closed topology and component volume signs.

These are CPU reconstructions using the existing ordinary FMA deformation path and saved native poses. They are **not 25 new GPU captures**, do not qualify external anatomical interfaces, and do not complete the five-minute integrated deliverable. The CPU predictor is not bit-identical to the GPU. Actual 16-second native 012 evidence remains in [1358](../native-anatomy-left-forearm-diagonal-m6-1358/README.md).

The grid includes eight parent-run poses, one posture-sensitivity pose, and eight poses each from the retained baseline and respiratory-intervention runs. Late poses extend to accepted step 155000 from the older 310-second runs. Those older runs supply kinematics only; they are not final native evidence for the repaired anatomy.

Source geometry, field coefficients and physical state were not changed by this audit. Both muscle identities, laterality and raw-source provenance remain those documented in 1358. The two new diagonal faces have pooled derived-face ancestry documented there. This is a mixed-source reference adult, not a measured individual.

Reproduce on the retained numi host:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/audit_left097_retained25_cpu_001.py
```

The script refuses to overwrite an existing result directory. Preserve this executed script and choose a new output directory in a successor copy for reruns. All source/helper/pose inputs and copied outputs are SHA-256 pinned in publication.json. Component volume signs are checked separately in component-volume-sign-audit.json; they are not a physical mass or density validation.

Right-forearm repair, back seams, quadratus/neighbor interfaces, and final integrated baseline/intervention recording remain open. No runtime code or performance claim is changed by this evidence increment.
"""
    (OUT/'README.md').write_text(text)
    require(all(sha(p)==d for p,d in external.items()),'source changed during publication')
    print(json.dumps({'status':status,'output':str(OUT),'files':len(copied),'pose_count':25,'component_checks':len(closures)}))
if __name__=='__main__':main()

