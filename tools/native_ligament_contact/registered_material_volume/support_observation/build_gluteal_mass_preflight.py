"""Attach the retained pelvic mass-partition preflight to the current complete native scene."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from bind_gluteal_receivers import digest, require


def build(current, owner, out):
    manifest=json.loads((current/'build.json').read_bytes())
    old_app=current/'app'
    old_exe=old_app/'numi-human-native'
    require(digest(old_exe)==manifest['executable'],'Unknown native parent')
    out.mkdir(exist_ok=True)
    name='numilab_human_myosim_visual_probe.mm'
    src=(old_app/name).read_text()
    header='#include "NumiHumanDigitCorticalContact.hpp"'
    hook='    const auto shift=[&](auto& p,std::uint32_t b) {'
    require(src.count(header)==1 and src.count(hook)==1,'Ambiguous native partition location')
    src=src.replace(header,header+'\n#include "NumiHumanPelvicMassPreflight.hpp"')
    src=src.replace(hook,'''    if (const char* ledger = std::getenv("NUMI_HUMAN_GLUTEAL_FAT_MASS_PREFLIGHT"); ledger && *ledger) {
        checkPelvicMassLedger(ledger, rebased, offset);
    }
'''+hook)
    anchor='        std::map<std::uint32_t,std::array<numiHumanCellShape::Summary,2>> shapeByMaterial;'
    require(src.count(anchor)==1,'Ambiguous accepted material audit location')
    src=src.replace(anchor,Path(__file__).with_name('gluteal_overlap_capture.inc').read_text()+chr(10)+anchor)
    txn='struct HumanTendonContinuumTransaction {'
    require(src.count(txn)==1,'Unknown continuum transaction')
    src=src.replace(txn,txn+chr(10)+'    std::vector<std::uint32_t> posteriorBedNodes;')
    for marker,inc in [('                                // The final compiled world now owns all physics','gluteal_contact_rows.inc'),
                       ('            std::cout << "resting_fem_skin_bed_terminal_history"','gluteal_contact_history.inc')]:
        require(src.count(marker)==1,'Unknown native bed history owner')
        src=src.replace(marker,Path(__file__).with_name(inc).read_text()+chr(10)+marker)
    (out/name).write_text(src)
    shutil.copyfile(owner,out/owner.name)
    for source_header in old_app.iterdir():
        if source_header.suffix in {'.hpp','.inc','.h'}:
            shutil.copyfile(source_header,out/source_header.name)
    compiled=[]
    for command in manifest['commands'][-2:]:
        args=[str(out/name) if a==str(old_app/name) else
              str(out/'numi-human-native.o') if a==str(old_app/'numi-human-native.o') else
              str(out/'numi-human-native') if a==str(old_exe) else a for a in command]
        # Preserve all source-local includes of the unchanged native owner.
        if str(out/name) in args:
            args[1:1]=['-iquote',str(old_app)]
        with (out/f'build-{len(compiled)}.log').open('w') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
        compiled.append(args)
    (out/'build.json').write_text(json.dumps({'parent':manifest['executable'],'library':manifest['library'],
        'source':digest(out/name),'mass_owner':digest(owner),'executable':digest(out/'numi-human-native'),
        'commands':compiled},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('current','owner','out'):
        parser.add_argument(key,type=lambda v:Path(v).resolve())
    build(**vars(parser.parse_args()))
