"""Bind qualified credits, recomputed P1 support and role witnesses to the repaired reference mesh."""
from pathlib import Path
import json,sys,os
import numpy as np
from compile_connected_joint import digest
from bind_connected_quadrature import run as bind
from replay_connected_allocation import run as replay
def run(job,h,p66,native,p1,credits,out):
    rd=lambda p:json.loads(p.read_bytes());out.mkdir()
    stage=out/'scene-material-p1';stage.mkdir();os.symlink(p1,out/'material-p1');os.symlink(job/'posed-app',out/'posed-app')
    cfg=rd(job/'registered-donor-cook/scene/launch.json');env=cfg['environment'];pr='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_';P='NUMI_H706_FRACTIONAL_'
    env.update(rd(p1/'result.json')['environment'])
    for k,fn in [('TETGEN_NODE','mesh.node'),('TETGEN_ELEMENTS','mesh.ele')]:env[pr+k]=str(native/fn);env[pr+k+'_SHA256']=digest(native/fn)
    historic=rd(job/'scene-material-p1/launch.json')['environment']
    for tag in ('SKIN','FAT'):
        for suffix in ('NPZ','SHA256'):
            k=P+'CARRIERS_'+tag+'_'+suffix;env[k]=historic[k]
    (stage/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    flows=out/'flows';flows.mkdir();src=job/'feasible-carriers';rec=rd(src/'result.json')
    s=np.load(credits/'repaired-credits.npz');packet=src/'FAT-physical-flow.npz';assert digest(packet)==rd(credits/'result.json')['source_flow']
    z=dict(np.load(packet));z['mass'][s['edge']]=s['mass'];keep=z['mass']>0
    dest=flows/packet.name;np.savez_compressed(dest,**{k:v[keep] for k,v in z.items()})
    rec['FAT'].update(flow_sha256=digest(dest),density_redistribution=digest(credits/'result.json'));os.link(src/'SKIN-physical-flow.npz',flows/'SKIN-physical-flow.npz')
    (flows/'result.json').write_text(json.dumps(rec,indent=2)+'\n')
    bind(h,p66,out,flows,out/'packets')
    owner=Path(rd(job/'native-unique-allocation/launch.json')['command'][0]).parent
    return replay(stage,out,flows,out/'packets',owner,out/'native-replay')
if __name__=='__main__':sys.exit(run(*(Path(v).resolve() for v in sys.argv[1:])))
