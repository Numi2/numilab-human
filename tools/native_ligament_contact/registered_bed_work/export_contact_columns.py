"""Capture all lower-envelope columns in the current native registered posture."""
import json,subprocess,sys
from pathlib import Path
from material_context import sha

def change(s,a,b):
    if s.count(a)!=1:raise ValueError('Unmatched bed material source: '+a)
    return s.replace(a,b)

def capture(source,owner,out):
    read=lambda p:json.loads(p.read_text())
    save=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
    build=out/'column-owner';dst=out/'column-source'
    ids=read(owner/'identities.json');cfg=read(source/'launch.json')
    if any(sha(owner/k)!=v for k,v in ids.items()):raise ValueError('Parent build changed')
    subprocess.run(['cp','-cR',owner,build],check=True);dst.mkdir()
    name='NumiHumanRestingVisual.hpp';p=build/name;s=p.read_text()
    anchor='                std::ofstream footprintReceipt;'
    s=change(s,anchor,'''                std::vector<double> fullAreas;
                std::ofstream complete;
                if (const char* output=std::getenv("NUMI_HUMAN_COMPLETE_BED_FOOTPRINT_CSV")) {
                    const double top=std::max_element(footprintPoints.begin(),footprintPoints.end(),
                        [](const auto& a,const auto& b){return a.z<b.z;})->z;
                    fullAreas=numiHumanBedFootprint::areas(footprintPoints,skin.indices,
                        std::vector<double>(footprintPoints.size(),top+1.0));
                    complete.open(output);
                    require(bool(complete),"cannot write registered bed columns");
                    complete<<"vertex,row,source_geometry_index,x_m,y_m,z_m,area_m2,skin_m,fat_m,thickness_source_vertex\\n"<<std::setprecision(17);
                }
'''+anchor)
    s=change(s,'if (tributaryAreas[v] == 0) continue;', 'if (tributaryAreas[v] == 0 && (fullAreas.empty() || fullAreas[v]==0)) continue;')
    anchor='                    if(footprintReceipt.is_open()){'
    s=change(s,anchor,'''                    if (!fullAreas.empty() && fullAreas[v]>0) {
                        complete<<v<<','<<row<<','<<regions[row].sourceGeometryIndex<<','
                            <<posedSkin[v].x<<','<<posedSkin[v].y<<','<<posedSkin[v].z<<','
                            <<fullAreas[v]<<','<<kWholeBodySkinThicknessM[nearest]<<','
                            <<kWholeBodyFatThicknessM[nearest]<<','<<kWholeBodySkinSourceVertices[nearest]<<'\\n';
                    }
                    if (tributaryAreas[v]==0) continue;
'''+anchor)
    p.write_text(s)
    commands=[[s.replace(str(owner)+'/',str(build)+'/') for s in cmd] for cmd in read(owner/'build-commands.json')]
    for cmd in commands[2:]:subprocess.run(cmd,cwd=Path(cfg['environment']['DYLD_LIBRARY_PATH']).parent,check=True)
    ids={k:sha(build/k) for k in set(ids)|{name}};save(build/'identities.json',ids);save(build/'build-commands.json',commands)
    cfg['environment']['NUMI_HUMAN_COMPLETE_BED_FOOTPRINT_CSV']=str(out/'registered-columns.csv')
    save(dst/'launch.json',cfg)
if __name__=='__main__':capture(*(Path(s).resolve() for s in sys.argv[1:]))
