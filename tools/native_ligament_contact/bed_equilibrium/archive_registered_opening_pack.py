"""Restore the native opening frame independently using its measured physical-parity receipt."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from evaluate_registered_bed_transfer import digest
run=Path(sys.argv[1]).resolve();pack=run/"initial-material-display.mrvpack"
identity=digest(pack)
if json.loads((run/"release-skin-parity.json").read_text())["initial_pack_sha256"]!=identity:raise ValueError("Unbound opening frame")
tag="human-closed-skin-diagnostic-packs-20261121";repo="Numi2/numilab-human"
asset="registered-loaded-opening-"+identity[:16]+".mrvpack.zst";gh="/opt/homebrew/bin/gh";zstd="/opt/homebrew/bin/zstd"
with tempfile.TemporaryDirectory(dir=run) as folder:
    folder=Path(folder);local=folder/asset
    subprocess.run([zstd,"-q","-1","-T1",str(pack),"-o",str(local)],check=True)
    if asset not in {x["name"] for x in json.loads(subprocess.check_output([gh,"release","view",tag,"-R",repo,"--json","assets"]))["assets"]}:
        subprocess.run([gh,"release","upload",tag,str(local),"-R",repo],check=True)
    recovered=folder/"verify";recovered.mkdir()
    subprocess.run([gh,"release","download",tag,"-R",repo,"-p",asset,"-D",str(recovered)],check=True)
    z=digest(local)
    if digest(recovered/asset)!=z:raise ValueError("Compressed archive differs")
    with subprocess.Popen([zstd,"-qdc",str(recovered/asset)],stdout=subprocess.PIPE) as proc:
        sha=hashlib.file_digest(proc.stdout,"sha256").hexdigest()
        if proc.wait() or sha!=identity:raise ValueError("Recovered opening frame differs")
    (run/"initial-display-recovery.json").write_text(json.dumps(dict(asset=asset,repository=repo,tag=tag,compressed_sha256=z,
        recovered_sha256=sha,restored_bytes=pack.stat().st_size,independent_download_verified=True,
        url=f"https://github.com/{repo}/releases/download/{tag}/{asset}"),indent=2)+"\n")
