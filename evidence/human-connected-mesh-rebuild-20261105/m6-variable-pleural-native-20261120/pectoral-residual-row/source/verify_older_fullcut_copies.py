"""Independently stream and verify the two previously restored GitHub archives."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess

gh = '/opt/homebrew/bin/gh'
home = Path.home()
folders = (
    home/'storage-cleanup-20261101-actual024-fullcut-archive-plan-001',
    home/'storage-cleanup-20261101-actual027-archive-prep-001/actual027',
)
results = []
for folder in folders:
    proof = json.loads((folder/'independent-restore-proof.json').read_text())
    local = folder/'full-cut-witness.csv.gz'
    archive = proof['archive']
    asset = f"repos/{proof['repository']}/releases/assets/{archive['asset_id']}"
    metadata = json.loads(subprocess.check_output([gh,'api',asset]))
    assert metadata['digest'] == f"sha256:{archive['compressed_sha256']}"
    assert metadata['size'] == archive['compressed_bytes'] == local.stat().st_size
    with local.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == archive['compressed_sha256']
    with subprocess.Popen([gh,'api','-H','Accept: application/octet-stream',asset],
                          stdout=subprocess.PIPE) as remote:
        checksum = hashlib.sha256()
        count = 0
        with gzip.GzipFile(fileobj=remote.stdout) as stream:
            while data := stream.read(4 << 20):
                checksum.update(data)
                count += len(data)
        assert remote.wait() == 0
    assert count == archive['uncompressed_bytes']
    assert checksum.hexdigest() == archive['uncompressed_sha256']
    results.append({
        'path': str(local),
        'release': proof['release_tag'],
        'repository': proof['repository'],
        'compressed_bytes': metadata['size'],
        'compressed_sha256': archive['compressed_sha256'],
        'restored_sha256': checksum.hexdigest(),
        'restored_bytes': count,
    })
    print(json.dumps(results[-1]),flush=True)
Path('older-fullcut-copy-recovery.json').write_text(json.dumps(results,indent=2)+'\n')
