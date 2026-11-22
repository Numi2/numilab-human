"""Keep verified release and accepted geometry assets outside the source checkout."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from emit_skin_source_registration import sha
scene=Path(sys.argv[1]).resolve()
out=scene/'recovery';out.mkdir(exist_ok=True)
repo='Numi2/numilab-human';tag='human-closed-skin-diagnostic-packs-20261121'
for path,src,label,key in [
    (scene/'coupled-3/initial-registered.mrvpack.zst',scene/'coupled-3/initial-pack-identity.json','reference','sha256'),
    (scene/'coupled-8-retry/terminal.mrvpack.zst',scene/'coupled-8-retry/native/accepted-geometry/step-8.receipt.json','accepted-8','pack_file_sha256')]:
    identity=json.loads(src.read_text())[key]
    name='connected-torso-'+label+'-'+identity[:14]+'.mrvpack.zst'
    compressed=sha(path)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        temp=Path(tmp);local=temp/name;local.hardlink_to(path)
        present=subprocess.check_output(['/opt/homebrew/bin/gh','release','view',tag,'-R',repo,'--json','assets'],text=True)
        if name not in {i['name'] for i in json.loads(present)['assets']}:
            subprocess.run(['/opt/homebrew/bin/gh','release','upload',tag,str(local),'-R',repo],check=True)
        target=temp/'download';target.mkdir()
        subprocess.run(['/opt/homebrew/bin/gh','release','download',tag,'-R',repo,'-p',name,'-D',str(target)],check=True)
        if sha(target/name)!=compressed:raise ValueError('Downloaded archive differs')
        with subprocess.Popen(['/opt/homebrew/bin/zstd','-q','-dc',str(target/name)],stdout=subprocess.PIPE) as decoder:
            h=hashlib.sha256();n=0
            for b in iter(lambda:decoder.stdout.read(2**20),b''):h.update(b);n+=len(b)
            if decoder.wait()!=0 or h.hexdigest()!=identity:raise ValueError('Unpacked native image differs')
    report=dict(repository=repo,release_tag=tag,asset=name,compressed_sha256=compressed,restored_sha256=identity,
                restored_bytes=n,independent_download_verified=True,url=f'https://github.com/{repo}/releases/download/{tag}/{name}')
    (out/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(label,n,identity,flush=True)
