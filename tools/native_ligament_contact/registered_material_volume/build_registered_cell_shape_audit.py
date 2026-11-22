"""Add geometric quality to the owning accepted-state certificate, using its compensated edges."""
import json,sys,subprocess
from pathlib import Path
from emit_skin_source_registration import sha
src,app,out=map(lambda v:Path(v).resolve(),sys.argv[1:])
previous=json.loads((src/'pleural-integration.json').read_text())
if sha(src/'numi-human-native')!=previous['executable_sha256']:raise ValueError('Unmatched native parent')
out.mkdir(exist_ok=False)
main='numilab_human_myosim_visual_probe.mm'
for f in src.iterdir():
    if f.is_file() and f.suffix in ('.hpp','.metal','.metalinc','.metallib'):(out/f.name).hardlink_to(f)
(out/'metalrobo').symlink_to((src/'metalrobo').resolve(),target_is_directory=True)
header=Path(__file__).with_name('NumiHumanMaterialCellShapeAudit.hpp')
(out/header.name).write_bytes(header.read_bytes())
s=(src/main).read_text()
def once(old,new):
    global s
    if s.count(old)!=1:raise ValueError('Unexpected certificate owner '+old[:70])
    s=s.replace(old,new)
s='#include "NumiHumanMaterialCellShapeAudit.hpp"\n#include <map>\n'+s
once('const auto det=[](const auto& nodes,const auto& t){','const auto det=[](const auto& nodes,const auto& t,double* shape=nullptr){')
start=s.index('const auto det=[](const auto& nodes,const auto& t,double* shape=nullptr)')
begin=s.index('            return b[0]*',start)
end=s.index(';',begin)+1
expr=s[begin:end]
once(expr,expr.replace('return ','const double determinant=')+'\n            if(shape) *shape=numiHumanCellShape::normalizedDeterminant(determinant,b,c,d);\n            return determinant;')
once('        std::uint32_t auditTetIndex=0u;', '''        std::uint32_t auditTetIndex=0u;
        const char* requestedShape=std::getenv("NUMI_HUMAN_REGISTERED_CELL_QUALITY");
        const bool measureShape=requestedShape && std::string_view(requestedShape)=="1";
        const auto firstCell=continuumTransaction->restingFEMRegionLabelFirstTetrahedron;
        const auto regionCells=continuumTransaction->restingFEMSourceTetRegionLabels.size();
        if(measureShape) require(regionCells>0 && firstCell<=a.femTopologyTetrahedra.size() &&
            regionCells<=a.femTopologyTetrahedra.size()-firstCell &&
            a.femTopologyTetrahedra.size()==continuumTransaction->restingAuditTetrahedra.size(),
            "current skin/fat quality requires accepted material and topology rows");
        std::map<std::uint32_t,std::array<numiHumanCellShape::Summary,2>> shapeByMaterial;''')
once('            const double restDet=det(z.femNodes,t);\n            const double currentDet=det(a.femNodes,t);','''            const bool selected=measureShape && auditTetIndex>=firstCell && auditTetIndex-firstCell<regionCells;
            double restShape=0.0,currentShape=0.0;
            const double restDet=det(z.femNodes,t,selected?&restShape:nullptr);
            const double currentDet=det(a.femNodes,t,selected?&currentShape:nullptr);
            if(selected){
                const auto& cell=a.femTopologyTetrahedra[auditTetIndex];
                require(cell.nodes.x==t[0] && cell.nodes.y==t[1] && cell.nodes.z==t[2] && cell.nodes.w==t[3],
                        "skin/fat certificate topology differs from accepted connectivity");
                auto& phases=shapeByMaterial[cell.identity.x];
                phases[0].observe(restDet,restShape,auditTetIndex-firstCell);
                phases[1].observe(currentDet,currentShape,auditTetIndex-firstCell);
            }''')
once('        for(unsigned i=0;i<a.femNodes.size();++i) {\n            const auto& p=a.femNodes[i].positionAndMass;', '''        for(const auto& [material,phases]:shapeByMaterial){
            for(unsigned k=0;k<2;++k){
                const auto& v=phases[k];
                std::cout<<std::setprecision(16)<<"registered_cell_shape phase="<<(k?"accepted":"reference")
                    <<" step="<<(k?stepCount:0u)<<" material="<<material<<" cells="<<v.cells
                    <<" nonfinite="<<v.nonfinite<<" inverted="<<v.inverted<<" volume_le_1e_18="<<v.belowVolume
                    <<" shape_lt_1e_5="<<v.belowShape<<" minimum_volume_m3="<<v.minimumVolume
                    <<" minimum_shape="<<v.minimumShape<<" minimum_shape_local_cell="<<v.minimumShapeRow<<std::endl;
            }
        }
        for(unsigned i=0;i<a.femNodes.size();++i) {
            const auto& p=a.femNodes[i].positionAndMass;''')
(out/main).write_text(s)
commands=[[arg.replace(str(src)+'/',str(out)+'/') for arg in cmd] for cmd in previous['commands']]
for cmd in commands:subprocess.run(cmd,cwd=app/'build-native',check=True)
(out/'registered-cell-audit-build.json').write_text(json.dumps(dict(parent_sha256=previous['executable_sha256'],header_sha256=sha(header),
    source_sha256=sha(out/main),binary_sha256=sha(out/'numi-human-native'),commands=commands),indent=2)+'\n')
