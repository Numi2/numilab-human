"""Reclaim blocks for exact retained-viewpack duplicates, preserving their paths."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
parents = [root.parent / f'resting-clearance-and-atrium-{i:03d}' for i in range(7,12)]
matches = defaultdict(list)
for parent in parents:
    for path in sorted(parent.glob('*/native/resting-human.mrvpack')):
        if path.stat().st_size != 296595456:
            continue
        with path.open('rb') as source:
            digest = hashlib.file_digest(source,'sha256').hexdigest()
        matches[digest].append(path)

reclaimed = []
for digest, paths in sorted(matches.items()):
    if len(paths) < 2:
        continue
    canonical = max(paths,key=lambda name:name.stat().st_nlink)
    for duplicate in paths:
        previous = duplicate.stat()
        if duplicate.samefile(canonical) or previous.st_nlink != 1:
            continue
        subprocess.run(['/usr/bin/cmp','-s',str(canonical),str(duplicate)],check=True)
        temp = duplicate.with_name(duplicate.name+'.dedup.tmp')
        os.link(canonical,temp)
        os.replace(temp,duplicate)
        assert duplicate.samefile(canonical)
        reclaimed.append(dict(sha256=digest,reference=str(canonical),duplicate=str(duplicate),bytes=previous.st_size))
result = {'reclaimed_logical_bytes':sum(row['bytes'] for row in reclaimed),
          'verified_duplicate_count':len(reclaimed),'replacements':reclaimed}
(root/'retained-viewpack-deduplication.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='replacements'}),flush=True)
