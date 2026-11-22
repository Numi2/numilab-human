"""Intersect authenticated one-hop regions across all physical cells of each flow recipient."""
import argparse,json
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha

def allowed(bits,rows,regions):
    word=np.right_shift(regions,6)
    return ((bits[rows,word] >> (regions%64).astype('u8'))&np.uint64(1))!=0
def main():
    p=argparse.ArgumentParser()
    for n in ('native','baseline_flow','candidate_flow','out'):p.add_argument(n,type=Path)
    args=p.parse_args();args.out.mkdir(exist_ok=True,parents=True);p1=args.native/'material/p1'
    state=json.load((p1/'current-paired-volume-state.json').open());labels=np.loadtxt(state['source330_identity']['source330_labels']['path'],delimiter=',',skiprows=1,dtype='i4');reg=labels[:,3]
    flow=np.load(args.baseline_flow);g=flow['group_for_cell'];bits=np.zeros((len(g),int(reg.max())//64+1),'u8');roles={}
    for role in ('skin','fat'):
        r=json.load((p1/(role+'-p1-receipt.json')).open());assert sha(Path(r['p1_one_hop_path']))==r['p1_one_hop_sha256']
        mesh=np.load(r['role_geometry_path']);child=mesh['combined_child_rows'];z=np.load(r['p1_one_hop_path']);off=z['cell_admissible_region_offsets'];ids=z['cell_admissible_region_ids']
        rows=np.repeat(child,np.diff(off));np.bitwise_or.at(bits,(rows,ids//64),np.left_shift(np.uint64(1),(ids%64).astype('u8')))
        roles[role]=dict(sha256=r['p1_one_hop_sha256'],children=len(child))
    assert np.all(bits.any(1));order=np.argsort(g,kind='stable');starts=np.r_[0,np.cumsum(np.bincount(g))];assert (np.diff(starts)>0).all()
    shared=np.bitwise_and.reduceat(bits[order],starts[:-1],axis=0);candidate=np.load(args.candidate_flow);err={}
    for nm,d in [('baseline',flow),('candidate',candidate)]:
        can=allowed(shared,d['group'],reg[d['source_parent']][d['carrier_row']]);bad=(~can)&(d['mass_flow_kg']>0)
        err[nm]=dict(ineligible_positive_edges=int(bad.sum()),ineligible_mass_kg=float(d['mass_flow_kg'][bad].sum()))
    out=args.out/'region-support.npz';np.savez_compressed(out,group_regions=shared,cell_regions=bits,source_region=reg[flow['source_parent']],current_state_sha256=sha(p1/'current-paired-volume-state.json'))
    (args.out/'audit.json').write_text(json.dumps(dict(roles=roles,initial_flow_check=err,region_support_sha256=sha(out)),indent=2)+'\n')
    print(err)
if __name__=='__main__':main()
