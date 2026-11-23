"""Certify both authored local frames and use their shared physical points when debiting rigid-body mass."""
import hashlib,json,os,struct,subprocess,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from rebind_coarse_skin import floats,block,donors
from run_cavity_motion import execute_native

def source_rows(folder,historical):
    header=(folder/'NumiHumanSelectedSkinReference.hpp').read_text();frame=floats(block(header,'frames')).reshape(-1,19)
    layer=(folder/'NumiHumanSelectedMaterialParent.hpp').read_text()
    delta=block((folder/'NumiHumanWholeBodySkinReduction.hpp').read_text(),'kWholeBodySkinMechanicalDonorLocalDeltas') if historical else block(header,'mechanical_local')
    nodes=np.concatenate((frame[:,1:4]+floats(delta).reshape(-1,3),floats(block(layer,'skin_local')).reshape(-1,3),floats(block(layer,'fat_local')).reshape(-1,3)))
    xyz=floats(block(header,'points')).reshape(-1,3)
    return np.tile(frame[:,0].astype('i4'),3),nodes,donors((folder/'NumiHumanLoadedSkinPoseData.inc').read_text(),'reference'),xyz

def frame_hash(b,p):
    h=hashlib.sha256()
    for donor,point in zip(b,p,strict=True):h.update(struct.pack('>I3d',int(donor),*point))
    return h.hexdigest()

def run(historical,base,control,out):
    cfg=json.loads((control/'launch.json').read_bytes());env=cfg['environment'];src=base/'app';out.mkdir();app=out/'app';app.mkdir()
    b,old,prior_frames, oldpoints=source_rows(historical,True)
    d,live,new_frames, newpoints=source_rows(src,False)
    parent=Path(env['NUMI_H706_FRACTIONAL_SOURCE330_NODES'])
    if digest(parent)!=env['NUMI_H706_FRACTIONAL_SOURCE330_NODES_SHA256']:raise ValueError('Source points pin differs')
    proof=np.loadtxt(parent,delimiter=',',skiprows=1,usecols=(9,10,11,12));same=np.array_equal
    if not same(b,d) or not same(oldpoints,newpoints) or not same(b,proof[:,0]) or not same(old,proof[:,1:]):raise ValueError('Historical node, donor, material geometry, or authored coordinate differs')
    by_body={};extreme=0
    for body in np.unique(b):
        sel=b==body;oo,R=prior_frames[body];nn,T=new_frames[body]
        inv=T.T@R
        if abs(np.linalg.det(inv)-1)>1e-12 or np.max(abs(inv.T@inv-np.eye(3)))>1e-12:raise ValueError('Non-rigid change of donor coordinates')
        miss=np.linalg.norm(old[sel]@R.T+oo-(live[sel]@T.T+nn),axis=1)
        by_body[int(body)]=float(max(miss));extreme=max(extreme,by_body[int(body)])
    if extreme>1e-6:raise ValueError('Paired material point differs by more than a micrometer')
    op,lp=frame_hash(b,old),frame_hash(d,live)
    result=dict(source_nodes=len(b),donors=len(by_body),source_local_sha256=op,registered_local_sha256=lp,maximum_world_discrepancy_m=extreme,maximum_coordinate_change_m=float(np.max(np.linalg.norm(live-old,axis=1))),donor_discrepancies_m=by_body,source_node_table=digest(parent),historical_headers={f:digest(historical/f) for f in ['NumiHumanSelectedSkinReference.hpp','NumiHumanSelectedMaterialParent.hpp','NumiHumanLoadedSkinPoseData.inc','NumiHumanWholeBodySkinReduction.hpp']},registered_headers={f:digest(src/f) for f in ['NumiHumanSelectedSkinReference.hpp','NumiHumanSelectedMaterialParent.hpp','NumiHumanLoadedSkinPoseData.inc']})
    (out/'frame-registration.json').write_text(json.dumps(result,indent=2)+'\n')
    header='NumiHumanPhysicalSkinReplacement.hpp';mm='numilab_human_myosim_visual_probe.mm'
    for f in src.iterdir():
        if f.name not in (header,mm) and f.suffix in ('.h','.hpp','.inc','.mm'):os.link(f.resolve(),app/f.name)
    s=(src/header).read_text()
    def exact(s,a,b):
        if s.count(a)!=1:raise ValueError('Unsupported source '+a[:70])
        return s.replace(a,b)
    s=exact(s,'struct RegisteredSkinFatReplacementAudit {','''struct RegisteredMaterialDonorChart {
    std::string_view historicalSHA256,registeredSHA256;
};
struct RegisteredSkinFatReplacementAudit {''')
    s=exact(s,'        bool& used,std::string& error) {','''        bool& used,std::string& error,
        const RegisteredMaterialDonorChart* chart,
        std::vector<V3>& registeredDonorLocals) {''')
    s=exact(s,'    for(const auto& [node,row]:byNode) {','''    spatial_mass_transport_detail::SHA256Builder historicalChart,registeredChart;
    if(chart)registeredDonorLocals.resize(count);
    for(const auto& [node,row]:byNode) {''')
    s=exact(s,'        preparedRows.push_back(local);','''        if(chart){
            const auto index=static_cast<std::size_t>(node-globalBase);
            if(row.donorBody!=parent.nodeFrameOwnerBodyIndices[index])
                return fail("source material point changed anatomical donor body");
            registeredDonorLocals[index]=row.localPosition;
            historicalChart.u32(row.donorBody);registeredChart.u32(row.donorBody);
            for(unsigned axis=0;axis<3;++axis){
                historicalChart.real(parent.nodeFrameOwnerLocalPoints[index][axis]);
                registeredChart.real(row.localPosition[axis]);
            }
            // Restore the independently authenticated source chart for the historical cook.
            // Its mass, source rows and native geometry remain subject to the exact comparisons below.
            local.localPosition=parent.nodeFrameOwnerLocalPoints[index];
        }
        preparedRows.push_back(local);''')
    s=exact(s,'    for(std::size_t node=0;node<count;++node) {\n        const auto& state=compiled.world.fem.nodes', '''    if(chart&&(historicalChart.finish()!=chart->historicalSHA256||
              registeredChart.finish()!=chart->registeredSHA256))
        return fail("material node local coordinates do not reproduce both registered authoring sources");
    for(std::size_t node=0;node<count;++node) {
        const auto& state=compiled.world.fem.nodes''')
    s=exact(s,'.massKg=row.massKg,.localPosition=row.donorLocalPosition});','.massKg=row.massKg,.localPosition=chart?registeredDonorLocals[row.parentNodeIndex]:row.donorLocalPosition});')
    a=s.index('[[nodiscard]] inline bool replaceLastPreparedRegisteredSkinFatUnpairedVolume(');u=s[a:]
    u=exact(u,'        std::string& error) {','''        std::string& error,const RegisteredMaterialDonorChart* chart=nullptr) {''')
    u=exact(u,'    bool usedCookedBaseline=false;','    bool usedCookedBaseline=false;\n    std::vector<V3> registeredDonorLocals;')
    u=exact(u,'capture,rawToCooked,usedCookedBaseline,error))return false;\n    const auto topology=bundle.topologyView();','''capture,rawToCooked,usedCookedBaseline,error,chart,registeredDonorLocals))return false;
    auto topology=bundle.topologyView();
    std::vector<ExplicitUnpairedVolumeSourceMassContribution> registeredSource;
    if(chart){
        if(!usedCookedBaseline||registeredDonorLocals.size()!=bundle.parent.referenceWorldPoints.size()){
            error="material-chart binding requires the verified source cook";return false;
        }
        registeredSource.assign(topology.parentNodeSourceMassContributions.begin(),topology.parentNodeSourceMassContributions.end());
        for(auto& credit:registeredSource){
            if(credit.parentNodeIndex>=registeredDonorLocals.size()||
               credit.donorBodyIndex!=bundle.parent.nodeFrameOwnerBodyIndices[credit.parentNodeIndex]||
               credit.donorLocalPosition!=bundle.parent.nodeFrameOwnerLocalPoints[credit.parentNodeIndex]){
                error="tissue debit is not an original registered source point";return false;
            }
            credit.donorLocalPosition=registeredDonorLocals[credit.parentNodeIndex];
        }
        topology.parentNodeSourceMassContributions=registeredSource;
    }''')
    (app/header).write_text(s[:a]+u)
    s=(src/mm).read_text();start=s.index('            require(replaceLastPreparedRegisteredSkinFatUnpairedVolume(\n                        source,wholeBodySkinObjectIndex,bundle,');scope=s[start:]
    scope=exact(scope,'installedSkin,installedFat,installedReport,replacementAudit,tissueError),','installedSkin,installedFat,installedReport,replacementAudit,tissueError,registeredSourceChart),')
    s=s[:start]+'''            const RegisteredMaterialDonorChart registeredCharts{"OLD_PIN","LIVE_PIN"};
            const auto* registeredSourceChart=selectedParent&&registeredFractionalMassSupportRequested?&registeredCharts:nullptr;
'''.replace('OLD_PIN',op).replace('LIVE_PIN',lp)+scope;(app/mm).write_text(s)
    commands=[[v.replace(str(src)+'/',str(app)+'/') for v in c] for c in json.loads((base/'build.json').read_bytes())['commands']]
    with (out/'build.log').open('w') as f:
        for c in commands:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
    exe=app/'numi-human-native';(out/'build.json').write_text(json.dumps(dict(commands=commands,prior_build=digest(base/'build.json'),source_frame_proof=digest(out/'frame-registration.json'),header=digest(app/header),application=digest(app/mm),executable=digest(exe)),indent=2)+'\n')
    job=out/'scene';job.mkdir();env={k:v.replace(str(control)+'/',str(job)+'/') for k,v in env.items()};cfg['argv'][0]=str(exe);cfg['argv'][4]=str(job/'native-traces');cfg['environment']=env;cfg['parent_launch']=digest(control/'launch.json');cfg['pins'][str(exe)]=digest(exe)
    (job/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(cfg['argv'],cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,job,minimum_free_bytes=4*2**30,timeout_s=400))
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
