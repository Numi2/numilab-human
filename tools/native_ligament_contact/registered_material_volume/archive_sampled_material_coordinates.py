"""Verify the four native node captures used to compare the two material registrations."""
import hashlib,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
from emit_skin_source_registration import sha
p=Path(sys.argv[1]);out=p/'recovery';files=[p/'accepted-cell-proof'/('step-eight-'+k+'.node') for k in ('reference','accepted')]+[p/'conditioned-inputs'/('endpoint-'+k+'.node') for k in ('reference','accepted')]
if len(sys.argv)>2:
    inputs=Path(sys.argv[2]).resolve()
    files=[inputs/('endpoint-'+phase+'.node') for phase in ('reference','accepted')]
    receipt=inputs.name+'-material-node-captures.json'
else:
    receipt='material-node-captures.json'
repo='Numi2/numilab-human';tag='human-closed-skin-diagnostic-packs-20261121';gh='/opt/homebrew/bin/gh'
pins={i.name:sha(i) for i in files}
label='registered-material-node-captures-'+hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()[:14]+'.zip'
with tempfile.TemporaryDirectory(dir=out) as tmp:
    root=Path(tmp);archive=root/label
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for i in files:z.write(i,i.name)
    assets=json.loads(subprocess.check_output([gh,'release','view',tag,'-R',repo,'--json','assets']))['assets']
    if label not in {a['name'] for a in assets}:subprocess.run([gh,'release','upload',tag,str(archive),'-R',repo],check=True)
    restored=root/'restored';restored.mkdir()
    subprocess.run([gh,'release','download',tag,'-R',repo,'-p',label,'-D',str(restored)],check=True)
    with zipfile.ZipFile(restored/label) as z:
        assert sorted(z.namelist())==sorted(pins)
        for name,digest in pins.items():
            with z.open(name) as src:
                if hashlib.file_digest(src,'sha256').hexdigest()!=digest:raise ValueError(name)
    (out/receipt).write_text(json.dumps(dict(repository=repo,tag=tag,asset=label,download_sha256=sha(restored/label),member_sha256=pins,url=f'https://github.com/{repo}/releases/download/{tag}/{label}',verified='independent download and complete member recovery'),indent=2)+'\n')
