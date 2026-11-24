"""Keep the registered plane, geometric admission and observation payload on the same frame."""
import json,re,struct
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from update_witnesses import reproject

def registered_header(code,data):
    if 'bed_height_m' not in data:return code
    constants={'minimumBedGap':data['minimum_gap_m'].hex(),'fixedBedHeight':data['bed_height_m'].hex(),
               'registeredBedContactNodes[]':'{'+','.join(str(i)+'u' for i in data['contact_nodes'])+'}'}
    for name,value in constants.items():
        code,count=re.subn(re.escape(name)+r'\s*=\s*[^;]*;',lambda _:name+' = '+value+';',code)
        if count!=1:raise ValueError('Ambiguous registered bed constant '+name)
    return code

def register_payload(cfg,fit):
    data=json.loads((fit/'fit.json').read_bytes())
    if 'bed_height_m' not in data:return
    argv=cfg['argv'];i=argv.index('--support-contact-payload')+1;old=Path(argv[i])
    source=json.loads(Path(data['contact_frame_receipt']).read_bytes())
    if digest(Path(data['contact_frame_receipt']))!=data['source_bed'] or digest(old)!=source['bed'] or cfg['pins'][str(old)]!=source['bed']:raise ValueError('Different physical support control')
    packet=bytearray(old.read_bytes());normal=np.array(data['bed_normal'])
    if packet[:8]!=b'NHCNT1\0\0' or struct.unpack_from('<6f',packet,56)!=(0.,0.,source['height_m'],*normal):raise ValueError('Invalid physical bed identity')
    height=data['bed_height_m'];struct.pack_into('<6f',packet,56,0.,0.,height,*normal)
    reproject(packet,height,normal)
    new=fit/'registered-bed.nhcnt'
    if new.exists() and new.read_bytes()!=packet:raise ValueError('Different saved physical bed')
    if not new.exists():new.write_bytes(packet)
    cfg['pins'][str(new)]=digest(new);argv[i]=str(new)
