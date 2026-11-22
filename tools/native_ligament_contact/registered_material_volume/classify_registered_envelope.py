"""Separate native eye-display surfaces from physical witnesses in a complete bank audit."""
import argparse,json,re
from pathlib import Path
from emit_skin_source_registration import sha

def main():
    parser=argparse.ArgumentParser()
    for key in ('audit','native_log','output'):parser.add_argument(key,type=Path)
    parser.add_argument('--window-receipt',type=Path)
    args=parser.parse_args()
    result=json.loads(args.audit.read_text())
    captured=re.findall(r'^resting_ocular_skin_windows=(\d+) derived_typed_ids=([\d,]+) posterior_backings=(\d+) backing_typed_ids=([\d,]+) source_vertices_unchanged=true source_index_prefix_unchanged=true physical_state_created=false ',args.native_log.read_text(),re.M)
    if len(captured)!=1:raise ValueError('Missing native eye-display ownership')
    windows,window_ids,backings,backing_ids=captured[0]
    win=tuple(map(int,window_ids.split(',')));back=tuple(map(int,backing_ids.split(',')))
    if len(win)!=int(windows) or len(back)!=int(backings) or len(set(win+back))!=len(win+back):raise ValueError('Ambiguous native display IDs')
    display_ids=set(win+back);display=0;ocular=0;physical=0;banks={}
    if args.window_receipt:
        document=json.loads(args.window_receipt.read_text())
        if not document['presentation_only'] or document['physical_state_created']:raise ValueError('Physical state in display receipt')
        for row in document['windows']+document['posterior_lens_backings']:
            d=row['derived_skin_identity']
            if d['semantic']!=51010 or d['stable_id'] in banks:raise ValueError('Unexpected display bank')
            banks[d['stable_id']]=row['derived_primitive_index']
        if set(banks)!=display_ids:raise ValueError('Display identities differ from native binding')
    for kind,stable,primitive,count in result['affected_banks']:
        if kind==51010 and stable in display_ids:
            if banks and banks[stable]!=primitive:raise ValueError('Display primitive identity mismatch')
            display+=count
        elif kind==51010 and 381<=stable<=397:ocular+=count
        else:physical+=count
    assert physical+display==result['nonocular_pairs'] and ocular==result['intersection_pairs_by_semantic'].get('ocular',0)
    record=dict(pack_sha256=result['pack_sha256'],audit_sha256=sha(args.audit),native_log_sha256=sha(args.native_log),
        window_receipt_sha256=sha(args.window_receipt) if args.window_receipt else None,
        checked_banks=sum(result['target_banks_by_semantic'].values()),physical_nonocular_pairs=physical,ocular_source_pairs=ocular,
        eye_presentation_only_pairs=display,bone_pairs=result['intersection_pairs_by_semantic'].get('51004',0))
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(record)
if __name__=='__main__':main()
