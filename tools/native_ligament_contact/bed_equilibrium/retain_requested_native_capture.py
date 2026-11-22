"""Retain the final native geometry using its independently written accepted receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def capture(run, volume, step):
    source=volume/'native/accepted-geometry'/f'step-{step}.mrvpack'
    while True:
        try:
            stream=source.open('rb');break
        except FileNotFoundError:
            if (run/'failure.json').exists():return
            if (run/'execution.json').exists() and not source.exists():return
            time.sleep(.15)
    with stream:
        receipt=run/'terminal.receipt.json'
        local=source.with_suffix('.receipt.json')
        while True:
            for file in (local,receipt):
                try:
                    state=json.loads(file.read_text())
                    if state['accepted_step']!=step:raise ValueError('wrong accepted state')
                    break
                except (FileNotFoundError,json.JSONDecodeError):pass
            else:
                if (run/'failure.json').exists():return
                time.sleep(.05);continue
            break
        (run/f'step-{step}.receipt.json').write_text(json.dumps(state,indent=2)+'\n')
        # Hold only the read descriptor while the RAM volume exists. Hash the
        # original bytes as they are compressed and verify decompression separately.
        tmp=run/f'step-{step}.mrvpack.zst.partial'
        final=run/f'step-{step}.mrvpack.zst'
        with tmp.open('xb') as out, subprocess.Popen(['/opt/homebrew/bin/zstd','-q','-3','-c'],stdin=subprocess.PIPE,stdout=out) as proc:
            h=hashlib.sha256()
            while chunk:=stream.read(4*1024**2):
                h.update(chunk);proc.stdin.write(chunk)
            proc.stdin.close()
            if proc.wait()!=0:raise OSError('compression failed')
    if h.hexdigest()!=state['pack_file_sha256']:raise ValueError('capture differs from accepted state')
    with subprocess.Popen(['/opt/homebrew/bin/zstd','-qdc',str(tmp)],stdout=subprocess.PIPE) as proc:
        restored=hashlib.file_digest(proc.stdout,'sha256').hexdigest()
        if proc.wait()!=0 or restored!=h.hexdigest():raise ValueError('compressed capture failed restore')
    os.replace(tmp,final)
    print(step,h.hexdigest(),final.stat().st_size,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('volume',type=Path);p.add_argument('step',type=int)
    a=p.parse_args();capture(a.run,a.volume,a.step)
