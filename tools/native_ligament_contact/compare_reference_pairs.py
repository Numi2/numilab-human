"""Compare exact terminal witnesses with the previously registered native entrance."""
import hashlib,json
from pathlib import Path
import sys

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def banks(doc):
    return {(r['semantic'],r['stable_id']):r for r in doc['surfaces']
            if not r['ocular_source_id'] and not r.get('ocular_presentation_only')}

def compare(prior, terminal):
    past=prior/'crossings-by-physical-scope.json'
    delta=prior/'complete-paired-pairs.json'
    change=json.loads(delta.read_text())
    if change['typed_audit_sha256']!=sha(past):raise ValueError('Paired reference identity mismatch')
    historic=banks(json.loads(past.read_text()));today=banks(json.loads(terminal.read_text()))
    if set(historic)!=set(today):raise ValueError('Scene surface roster differs')
    first=set();last=set()
    for r in change['banks']:
        key=tuple(r['key']);a,b=historic[key],today[key]
        if a['target_faces']!=b['target_faces']:raise ValueError(f'Primitive face inventory changed in {key}')
        after=set(map(tuple,a['triangle_pairs']))
        start=after.difference(map(tuple,r['terminal_only_pairs']))|set(map(tuple,r['initial_only_pairs']))
        if len(start)!=r['initial']:raise ValueError('Cannot recover reference pair identities')
        first.update(key+x for x in start);last.update(key+tuple(x) for x in b['triangle_pairs'])
    return dict(prior_delta_sha256=sha(delta),prior_terminal_sha256=sha(past),current_terminal_sha256=sha(terminal),
                scope='Stable bank and face-index pairs in the former registered release entrance and this accepted terminal capture. Two distinct runs and skeleton hashes.',
                initial_reference=len(first),current=len(last),shared=len(first & last),only_current=sorted(last-first),only_reference=sorted(first-last))

if __name__=='__main__':
    prior,terminal,output=map(Path,sys.argv[1:])
    output.write_text(json.dumps(compare(prior,terminal),indent=2)+'\n')
