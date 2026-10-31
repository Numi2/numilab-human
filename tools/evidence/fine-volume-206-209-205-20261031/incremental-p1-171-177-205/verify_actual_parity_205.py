#!/usr/bin/env python3
"""Verify a bounded changed-row bundle against authenticated 171 rows."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import incremental_p1_reuse_205 as reuse

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_selected(path: Path, requested: set[int]):
    found={}
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            row=int(r['child_tet_row'])
            if row in requested:
                if row in found: raise ValueError(f'duplicate CSV row {row} in {path}')
                found[row]=r
    if set(found)!=requested: raise ValueError(f'{path} does not cover requested rows')
    return found

def verify(base_dir: Path, subset_dir: Path, request_path: Path):
    request=list(csv.DictReader(request_path.open(newline='',encoding='utf-8')))
    rows=[int(r['candidate_child_tet_row']) for r in request]
    if not rows or rows!=sorted(set(rows)):
        raise ValueError('request rows must be unique, sorted, and nonempty')
    base_receipt=json.loads((base_dir/'receipt.json').read_text())
    fresh_receipt=json.loads((subset_dir/'receipt.json').read_text())
    if base_receipt.get('full_child_domain_export') is not True or base_receipt.get('physical_admission') is not False:
        raise ValueError('base 171 receipt is not a full source-only export')
    if fresh_receipt.get('full_child_domain_export') is not False or fresh_receipt.get('physical_admission') is not False:
        raise ValueError('changed-row receipt must remain an incomplete source-only subset')
    for k,p in [('labels','child.labels.csv'),('parent_row_map','child.labels.csv.source.csv'),
                ('p1_anchor_projection_csv','p1-anchor-projection.csv'),('p1_anchor_csr','p1-anchor.npz')]:
        for rec,dir_ in [(base_receipt,base_dir),(fresh_receipt,subset_dir)]:
            if rec['outputs'][k]['sha256']!=sha(dir_/p): raise ValueError(f'{k} output hash mismatch')
    array_checks=0; csr_checks=0
    with np.load(base_dir/'p1-anchor.npz',allow_pickle=False) as b, np.load(subset_dir/'p1-anchor.npz',allow_pickle=False) as f:
        if b['schema'].item()!=f['schema'].item(): raise ValueError('P1 schema mismatch')
        if not np.array_equal(f['child_tet_row'],np.asarray(rows,dtype=np.int64)):
            raise ValueError('subset P1 row IDs differ from request')
        for key in sorted(reuse.CELL_KEYS):
            if key=='p1_support_cell_row':
                if not np.array_equal(f[key],np.full(len(rows),-1,dtype=f[key].dtype)):
                    raise ValueError('new-cell support row sentinel is not -1')
            elif not np.array_equal(f[key],b[key][rows]):
                raise ValueError(f'P1 cell data differs for {key}')
            array_checks+=1
        for off,payloads in reuse.CSR_GROUPS.items():
            for local,row in enumerate(rows):
                blo,bhi=map(int,b[off][row:row+2]); flo,fhi=map(int,f[off][local:local+2])
                for key in payloads:
                    if not np.array_equal(b[key][blo:bhi],f[key][flo:fhi]):
                        raise ValueError(f'P1 CSR differs at {off}/{key}, row {row}')
                    csr_checks+=1
    csv_fields_compared=0
    for name in ('child.labels.csv','child.labels.csv.source.csv','p1-anchor-projection.csv'):
        requested=set(rows); a=read_selected(base_dir/name,requested); z=read_selected(subset_dir/name,requested)
        for row in rows:
            if a[row]!=z[row]: raise ValueError(f'{name} data differs at row {row}')
            csv_fields_compared+=len(a[row])
    return {
        'schema':'numi.native-domain-217.p1-reference-anchor.changed-row-parity.v1',
        'status':'PASS_BOUNDED_171_ROW_PARITY', 'physical_admission':False,
        'mass_transport':'NOT_RUN','native_steps':0,
        'row_count':len(rows),'rows':rows,
        'p1_cell_fields_compared':array_checks,'p1_csr_payload_slices_compared':csr_checks,
        'csv_field_values_compared':csv_fields_compared,
        'csv_outputs_compared':['child.labels.csv','child.labels.csv.source.csv','p1-anchor-projection.csv'],
        'source_sentinel':'p1_support_cell_row=-1 for changed rows; other P1 row data matches old 171 for this unchanged-mesh fixture',
        'base_receipt_sha256':sha(base_dir/'receipt.json'),
        'fresh_receipt_sha256':sha(subset_dir/'receipt.json'),
        'request_sha256':sha(request_path),
        'outputs':{name:{'path':str(subset_dir/name),'sha256':sha(subset_dir/name),'bytes':(subset_dir/name).stat().st_size}
                   for name in ('child.labels.csv','child.labels.csv.source.csv','p1-anchor-projection.csv','p1-anchor.npz')},
    }

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',type=Path,required=True); ap.add_argument('--subset-dir',type=Path,required=True); ap.add_argument('--request',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); result=verify(a.base_dir,a.subset_dir,a.request); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True))
