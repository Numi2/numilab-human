"""Recover the original, hash-pinned anatomical surfaces for the registered shoulder."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile
import zlib
from compile_connected_joint import digest

TARGET={75,79,83,85,87,89,153,155}


def pull(url,a,b):
    data=subprocess.check_output(['curl','-fsSL','--retry','2','--connect-timeout','8','--max-time','30','-r',f'{a}-{b}',url])
    if len(data)!=b-a+1:raise ValueError('Server did not return the requested ZIP member range')
    return data


def recover(cfg_file,lock_file,out,local=None):
    spec=json.loads(lock_file.read_bytes())['sources']['bodyparts3d_4']
    cfg=json.loads(cfg_file.read_bytes())
    manifest=next(Path(f)for f in cfg['pins']if f.endswith('fullbody-muscle-surfaces.manifest.json') and Path(f).is_file())
    if digest(manifest)!=cfg['pins'][str(manifest)]:raise ValueError('Changed tissue identity manifest')
    m=json.loads(manifest.read_bytes())['source']
    rows={r['stable_id']:r for r in m['surfaces'] if r['stable_id'] in TARGET}
    if set(rows)!=TARGET:raise ValueError('Missing shoulder source')
    archive='isa_BP3D_4.0_obj_99.zip';url=spec['base_url']+archive
    pin=next(z['sha256'] for z in m['bodyparts']['archives'] if z['file']==archive)
    out.mkdir(exist_ok=True)
    if local and local.exists():
        if digest(local)!=pin:raise ValueError('Changed downloaded archive')
        with zipfile.ZipFile(local)as z:data={sid:z.read(row['member']) for sid,row in rows.items()}
    else:
        head=subprocess.check_output(['curl','-fsSIL','--max-time','20',url],text=True)
        length=int([l.split(':',1)[1].strip()for l in head.splitlines()if l.lower().startswith('content-length:')][-1])
        tail=pull(url,length-65557,length-1);i=tail.rfind(b'PK\x05\x06');hdr=struct.unpack_from('<4s4H2LH',tail,i)
        _,disk,start,n1,n2,count,at,comment=hdr
        if disk or start or n1!=n2 or at+count+22+comment!=length:raise ValueError('Unsupported central directory')
        central=pull(url,at,at+count-1);mini=central+struct.pack('<4s4H2LH',hdr[0],0,0,n1,n2,count,0,0)
        with zipfile.ZipFile(io.BytesIO(mini)) as zi:info={sid:zi.getinfo(row['member'])for sid,row in rows.items()}
        def member(entry):
            sid,v=entry;a=v.header_offset;header=pull(url,a,a+29)
            if header[:4]!=b'PK\x03\x04':raise ValueError('ZIP entry signature')
            size,extra=struct.unpack_from('<2H',header,26);k=a+30+size+extra
            data=pull(url,k,k+v.compress_size-1)
            if v.compress_type!=zipfile.ZIP_DEFLATED:raise ValueError('Unsupported source compression')
            return sid,zlib.decompress(data,-15)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:data=dict(ex.map(member,info.items()))
    for sid,content in data.items():
        row=rows[sid]
        if hashlib.sha256(content).hexdigest()!=row['member_sha256']:raise ValueError('Source member hash mismatch')
        (out/Path(row['member']).name).write_bytes(content)
    (out/'recovery.json').write_text(json.dumps(dict(url=url,archive_sha256=pin,manifest=digest(manifest),
        source_license=spec['license'],attribution=spec['attribution'],members={k:{a:r[a]for a in ('member','member_sha256','label')}for k,r in rows.items()}),indent=2)+'\n')
if __name__=='__main__':recover(*(Path(a).resolve() for a in sys.argv[1:]))
