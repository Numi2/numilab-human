"""Match exact whole-skin witnesses across two accepted, source-identical native endpoints."""
import csv,hashlib,json,sys
from pathlib import Path

def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):
    report=json.loads((p/'exact-skin-surface.json').read_text());receipt=json.loads((p/'terminal.receipt.json').read_text())
    if report["accepted_pack_sha256"]!=receipt["pack_file_sha256"] or report["accepted_step"]!=receipt["accepted_step"]:
        raise ValueError("Endpoint capture and crossing report do not agree")
    banks={(s["semantic"],s["stable_id"],s["primitive"]):s for s in report["surfaces"]}
    if len(banks)!=len(report['surfaces']):raise ValueError('Repeated anatomical primitive identity')
    return receipt,report,banks
def changes(x,y):
    return dict(initial=len(x),terminal=len(y),retained=len(x&y),appeared=len(y-x),cleared=len(x-y))
def pairs(row):
    p={tuple(v) for v in row["triangle_pairs"]}
    if len(p)!=row["count"]:raise ValueError('The exact report does not retain every pair')
    return p
def compare(initial,terminal,out):
    ri,si,bi=read(initial);rf,sf,bf=read(terminal)
    if ri['base_pack_file_sha256']!=rf['base_pack_file_sha256'] or bi.keys()!=bf.keys():raise ValueError('Different registered geometry')
    cfg=json.loads((initial/'launch.json').read_text())
    args=cfg['argv'];tissue=Path(args[args.index('--soft-tissue-payload')+1]);manifest=tissue.with_suffix('.manifest.json')
    m=json.loads(manifest.read_text())
    if m["payload"]["sha256"]!=digest(tissue):raise ValueError("Muscle source identity mismatch")
    labels={r['stable_id']:r for r in m['source']['surfaces']}
    result={"base_pack_sha256":ri["base_pack_file_sha256"],"initial_step":si["accepted_step"],"terminal_step":sf["accepted_step"],
            "initial_audit_sha256":digest(initial/'exact-skin-surface.json'),"terminal_audit_sha256":digest(terminal/'exact-skin-surface.json'),
            "self":changes(pairs(si['self_intersections']),pairs(sf['self_intersections'])),"nonocular":{},'banks':[]}
    for identity,a in bi.items():
        b=bf[identity]
        if a['target_faces']!=b['target_faces'] or a['ocular_source_id']!=b['ocular_source_id']:raise ValueError('Anatomical target changed')
        count=changes(pairs(a),pairs(b));semantic,stable,primitive=identity
        row=dict(semantic=semantic,stable_id=stable,primitive=primitive,ocular=a['ocular_source_id'],**count)
        if semantic==51005:
            label=labels[stable]
            if 'triangle_count' in label and label['triangle_count']!=a['target_faces']:raise ValueError('Muscle geometry identity mismatch')
            row['source_name']=label.get('label') or label.get('name') or label['source_name']
        if not row['ocular']:
            for k,v in count.items():result['nonocular'][k]=result['nonocular'].get(k,0)+v
        result['banks'].append(row)
    result["inspected_targets"]=len(bi)
    result["targets_with_nonocular_intersections_by_first_step"]=sum(not x["ocular"] and x["initial"]>0 for x in result["banks"])
    result["targets_with_nonocular_intersections_at_terminal"]=sum(not x["ocular"] and x["terminal"]>0 for x in result["banks"])
    result['banks'].sort(key=lambda r:-r['terminal'])
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=="__main__":compare(*map(Path,sys.argv[1:]))
