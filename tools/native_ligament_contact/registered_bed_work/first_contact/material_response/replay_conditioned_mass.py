"""Replay the admitted source quadrature with current geometry, P1 support, recipients and witnesses."""
import json
from pathlib import Path
import sys
from register_material_surfaces import sha
from replay_material_mass import run as replay


def run(control,mesh,p1,carriers,witness,flows,solver,owner,out):
    j=json.loads(control.read_bytes())
    e=j['environment'].copy();p='NUMI_H706_FRACTIONAL_'
    S=json.loads((p1/'receipt.json').read_bytes())
    C=json.loads((carriers/'carriers.json').read_bytes())
    W=json.loads((witness/'witnesses.json').read_bytes())
    def pair(name,file,h):
        if sha(Path(file))!=h:raise ValueError('Input changed '+name)
        e[p+name]=file
    for tag in ('SKIN','FAT'):
        s=S[tag.lower()];c=C[tag];w=W[tag]
        pair('P1_'+tag+'_NPZ',s['p1_one_hop_path'],s['p1_one_hop_sha256'])
        e[p+'P1_'+tag+'_SHA256']=s['p1_one_hop_sha256']
        pair('CARRIERS_'+tag+'_NPZ',c['path'],c['sha256'])
        e[p+'CARRIERS_'+tag+'_SHA256']=c['sha256']
        pair('CARRIER_WITNESS_'+tag+'_CSV',w['path'],w['sha256'])
        e[p+'CARRIER_WITNESS_'+tag+'_SHA256']=w['sha256']
        if sha(Path(s['state_path']))!=s['state_sha256'] or sha(Path(s['role_geometry_path']))!=s['role_geometry_sha256']:
            raise ValueError('P1 state or geometry changed')
        e[p+'STATE_SHA256']=s['state_sha256']
        e[p+'GEOMETRY_'+tag+'_SHA256']=s['role_geometry_sha256']
    launch=out.with_suffix('.inputs.json')
    launch.write_text(json.dumps({'environment':e,'source_launch':sha(control),'conditioning_flow_audit':sha(flows/'exact-audit.json')},indent=2)+'\n')
    return replay(launch,mesh,flows,solver,owner,out)


if __name__=='__main__':sys.exit(run(*(Path(a).resolve() for a in sys.argv[1:])))
