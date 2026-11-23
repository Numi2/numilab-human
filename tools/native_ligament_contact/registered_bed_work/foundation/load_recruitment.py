"""Recruit MyoSim at the same frozen pose with the measured material reaction."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
from material_context import sha
from export_contact_columns import change
def build(base,out,run,source):
    ids=json.loads((base/'identities.json').read_text())
    if any(sha(base/k)!=v for k,v in ids.items()):raise ValueError('parent changed')
    r=json.loads((run/'terminal.receipt.json').read_text()); launches=json.loads((run/'launch.json').read_text())
    if r['accepted_step']!=1 or sha(source/'foundation.json')!=launches['pins'][str(source/'foundation.json')]:raise ValueError('stale opening')
    terminal,=[json.loads(l.split('=',1)[1]) for l in (run/'stdout.log').read_text().splitlines() if l.startswith('stand_terminal_state=')]
    q=np.array(terminal['initial_q'],dtype='f4')
    tab=run.with_name(run.name+'-source')/'witness.json.opening.csv'
    a=np.loadtxt(tab,delimiter=',',skiprows=1);g=a[:,1:-1].sum(axis=0);reaction=float(a[:,-1].sum())
    expected=json.loads((source/'foundation.json').read_text())['projected_skin_reaction_N']
    if (a.shape!=(38,len(q)+1) or not np.isfinite(a).all() or np.any(a[:,-1]<0) or
        not np.array_equal(a[:,0],np.arange(38)) or abs(reaction-expected)>.01 or abs(g[2]-reaction)>.001 or max(abs(g[:2]))>.001):
        raise ValueError('invalid normal load or row identity')
    subprocess.run(['cp','-cR',base,out],check=True)
    def vec(a):return ','.join(float(x).hex() for x in a)
    n='NumiHumanRegisteredRecruitment.inc'
    (out/n).write_text('static constexpr double openingQ[]={'+vec(q)+'};\nstatic constexpr double openingLoad[]={'+vec(g)+'};\n')
    n2='numilab_human_myosim_visual_probe.mm';f=out/n2;s=f.read_text()
    anchor='    double sourceDynamicForceParityMaximumNewtons = 0.0;'
    start=s.index('    double sourceDynamicForceParityMaximumNewtons = 0.0;',s.index('std::optional<CapturedWeightedSupportWitnesses> weightedCapture;'))
    prefix,s=s[:start],s[start:]
    s=s.replace(anchor,'''    #include "NumiHumanRegisteredRecruitment.inc"
    const bool useFoundationLoad=std::getenv("NUMI_HUMAN_RECRUIT_AT_FOUNDATION_LOAD")!=nullptr;
    std::optional<metalrobo::NumiHumanStaticExternalGeneralizedLoad> measuredBed;
    std::vector<double> canonicalLoadQ;
    if(useFoundationLoad) {
        require(!useStaticCache && useWeightedSkinSupport && !useFEMSkinSupport &&
                restingReleaseInitialization && !applySelectedActivationIncrement &&
                std::getenv("NUMI_HUMAN_RESOLVED_BED_WITNESS") &&
                weightedCompileQ.size()==std::size(openingQ) && model.world.nv==std::size(openingLoad),
                "unexpected recruitment pose or support owner");
        for(size_t i=0;i<weightedCompileQ.size();++i)
            require(static_cast<float>(weightedCompileQ[i])==static_cast<float>(openingQ[i]),
                    "foundation recruitment load belongs to a different pose");
        canonicalLoadQ=weightedCompileQ;
        const auto projected=metalrobo::projectNumiHumanJointEqualities(jointEqualities.payload.records,canonicalLoadQ);
        require(projected.succeeded() && packMetalConfiguration(canonicalLoadQ)==packMetalConfiguration(weightedCompileQ),
                "equality projection changed native opening coordinates");
        measuredBed=metalrobo::NumiHumanStaticExternalGeneralizedLoad{
            openingLoad,canonicalLoadQ,metalrobo::numiHumanStaticSupportModelFingerprint(model,0)};
    }
'''+anchor,1)
    s=prefix+s
    s=change(s,'            useWeightedSkinSupport && !useFEMSkinSupport\n                ?', '            useWeightedSkinSupport && !useFEMSkinSupport && !useFoundationLoad\n                ?')
    s=change(s,'            equilibriumGravityScale,\n            restingSupportedInitialization\n','            equilibriumGravityScale,\n            restingSupportedInitialization, measuredBed ? &*measuredBed : nullptr\n')
    s=change(s,'    if (useFEMSkinSupport) {\n        require(restingReleaseInitialization','    if (useFEMSkinSupport || useFoundationLoad) {\n        require(restingReleaseInitialization')
    s=change(s,'std::cout<<"resting_FEM_bed_release_initialization=prepared"','std::cout<<(useFoundationLoad?"registered_foundation_recruitment=prepared":"resting_FEM_bed_release_initialization=prepared")')
    s=change(s,'<<" support_owner=existing_Matter_FEM_contact"<<std::endl;', '<<(useFoundationLoad?" support_owner=registered_skin_quadrature":" support_owner=existing_Matter_FEM_contact")<<std::endl;')
    f.write_text(s)
    cmds=[[v.replace(str(base)+'/',str(out)+'/') for v in c] for c in json.loads((base/'build-commands.json').read_text())]
    cwd=Path(launches['environment']['DYLD_LIBRARY_PATH']).parent
    for cmd in cmds[2:]:subprocess.run(cmd,cwd=cwd,check=True)
    (out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    ids[n]=''
    (out/'identities.json').write_text(json.dumps({k:sha(out/k) for k in ids},indent=2)+'\n')
    (source/'opening-muscle-load.json').write_text(json.dumps(dict(force_table=sha(tab),launch=sha(run/'launch.json'),forces_N=reaction,expected_N=expected,force_vector=g.tolist(),initial_q=q.astype(float).tolist(),body_pose_fingerprint=r['base_pack_file_sha256'],scope='Only initialization consumes the stored generalized force. No runtime support impulse is seeded.'),indent=2)+'\n')
if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
