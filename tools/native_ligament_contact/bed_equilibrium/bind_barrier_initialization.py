"""Add matched IPC-band coordinates to the registered Human initializer, preserving the physical plane."""
from pathlib import Path
import json

def bind(out,base,band,pose):
    def replace_file(name,old,new):
        p=out/name;s=p.read_text()
        assert s.count(old)==1,(name,old)
        # Staged owner sources can be hard-linked to an older native build.
        p.unlink();p.write_text(s.replace(old,new))
    width=band['band_m'];gap=band['minimum_m']
    assert 0<len(band['nodes'])==len(band['gaps']) and 0<gap<=max(band['gaps'])<width
    assert json.loads((pose/'loaded-registration.json').read_text())['min_outer_height_m']==gap
    name='NumiHumanLoadedSkinPose.hpp'
    replace_file(name,'if(minimum!=0.)',f'if(minimum!={gap.hex()})')
    replace_file(name,'touch+=p.at(n)[2]==0.;',f'touch+=p.at(n)[2]<{width.hex()};')
    replace_file(name,'require(touch>0,',f'require(touch=={len(band["nodes"])},')
    replace_file(name,'<<" outer_contact_nodes="',f'<<" fixed_bed_offset_m=0 minimum_skin_gap_m="<<{gap.hex()}<<" ipc_width_m="<<{width.hex()}<<" outer_contact_nodes="')
    target='''        const double targetGap=(barrierRelease&&std::string_view(barrierRelease)=="1") ||
            trialPlane ? source.contactSlop : 0.0;'''
    new=f'''        const bool loadedBed=numiLoadedSkin::enabled();
        if(loadedBed)require(!trialPlane && !(barrierRelease && std::string_view(barrierRelease)=="1") &&
            float(source.contactSlop)=={width.hex()},"registered IPC pose requires its source contact bandwidth and fixed bed");
        const double targetGap=loadedBed ? {gap.hex()} : ((barrierRelease&&std::string_view(barrierRelease)=="1") ||
            trialPlane ? source.contactSlop : 0.0);'''
    namespace=next(v.split()[1].strip('{') for v in (out/name).read_text().splitlines() if v.startswith('namespace '))
    replace_file('numilab_human_myosim_visual_probe.mm',target,new.replace('numiLoadedSkin::',namespace+'::'))
