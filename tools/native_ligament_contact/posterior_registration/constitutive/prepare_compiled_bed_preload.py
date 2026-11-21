"""Bind the existing condensed bed law to fresh, fixed-pose recruitment."""
import os
from pathlib import Path
import sys
source,target=map(Path,sys.argv[1:])
s=source.read_text()
def change(a,b):
    global s
    if s.count(a)!=1:raise ValueError(a)
    s=s.replace(a,b)
mark='    if (staticCacheExists) {\n        compiledActivation = readStandActivationCache('
block='''    // The material coordinate is negative when compliant tissue is compressed.
    // At zero initial velocity each section carries -gap/C. Recruit against
    // that single physical load on the same captured Jacobian as the GPU row.
    std::vector<double> sectionLoads, sectionGaps, bedDofForce, bedReferenceQ;
    std::optional<metalrobo::NumiHumanStaticExternalGeneralizedLoad> bedRecruitment;
    if (!localBedSections.empty() && weightedCapture && preserveRestingInitialPosture &&
            !useFEMSkinSupport && !restingSupportedInitialization) {
        const auto& supports=weightedCapture->staticSupports;
        require(supports.size()==localBedSections.size() && !supports.empty(),
                "compliant bed recruitment row count");
        bool loaded=false;
        for(const auto& support:supports) loaded |= weightedSupportPlaneGap(support)<0;
        if(loaded){
            const std::size_t nv=model.articulations.front().nv;
            const auto fingerprint=metalrobo::numiHumanStaticSupportModelFingerprint(model,0);
            bedDofForce.assign(nv,0.0);
            double total=0.0,maxFraction=0.0;
            for(std::size_t row=0;row<supports.size();++row){
                const auto& s=supports[row];
                const auto& w=*s.weightedWitness;
                const auto& section=localBedSections[row];
                require(w.sourceContactIndex==row && w.modelFingerprint==fingerprint &&
                        w.pointJacobian.size()==3*nv && w.referenceQ==weightedCompileQ,
                        "compliant bed preload lost its matched skin witness");
                const double gap=weightedSupportPlaneGap(s);
                const float c=section.skinComplianceMPerN+section.fatComplianceMPerN;
                const double x=std::max(0.0,-gap);
                require(std::isfinite(c) && c>0 && std::isfinite(section.stockThicknessM) &&
                        section.stockThicknessM>0 && x<section.stockThicknessM,
                        "compliant bed construction exceeds registered tissue thickness");
                const double force=x/c;
                sectionGaps.push_back(gap);sectionLoads.push_back(force);
                total+=force;maxFraction=std::max(maxFraction,x/section.stockThicknessM);
                for(std::size_t j=0;j<nv;++j)
                    for(std::size_t k=0;k<3;++k) bedDofForce[j]+=force*s.normal[k]*w.pointJacobian[k*nv+j];
            }
            bedReferenceQ=weightedCompileQ;
            const auto projected=metalrobo::projectNumiHumanJointEqualities(jointEqualities.payload.records,bedReferenceQ);
            require(projected.succeeded() && packMetalConfiguration(bedReferenceQ)==packMetalConfiguration(weightedCompileQ),
                    "exact recruitment projection changes the actual captured Metal pose");
            bedRecruitment.emplace(metalrobo::NumiHumanStaticExternalGeneralizedLoad{
                    bedDofForce,bedReferenceQ,fingerprint});
            std::cout<<std::setprecision(12)<<"resting_bed_section_preload=constitutive rows="
                     <<supports.size()<<" compression_fraction_max="<<maxFraction<<" total_normal_N="<<total
                     <<" runtime_external_force_added=false"<<std::endl;
        }
    }
'''
change(mark,block+mark)
change('            useWeightedSkinSupport && !useFEMSkinSupport\n                ? std::span<const metalrobo::NumiHumanStaticSupportContact>(',
       '            useWeightedSkinSupport && !useFEMSkinSupport && !bedRecruitment\n                ? std::span<const metalrobo::NumiHumanStaticSupportContact>(')
mark='            restingSupportedInitialization\n        );\n    }\n    if (preserveRestingInitialPosture && !useWeightedSkinSupport)'
change(mark,'''            restingSupportedInitialization,
            bedRecruitment ? &*bedRecruitment : nullptr
        );
    }
    if(bedRecruitment){
        require(compiledActivation.supportNormalForce.empty() && compiledActivation.supportWitnessIdentity.empty() &&
                compiledActivation.q==bedReferenceQ &&
                std::all_of(compiledActivation.generalizedSupportForce.begin(),compiledActivation.generalizedSupportForce.end(),
                            [](double f){return f==0.0;}),
                "compliant bed load belongs to a different pose or duplicates support");
        compiledActivation.generalizedSupportForce=bedDofForce;
        compiledActivation.supportNormalForce=sectionLoads;
        compiledActivation.supportPlaneGapMeters=sectionGaps;
        for(const auto& support:weightedCapture->staticSupports){
            const auto& w=*support.weightedWitness;
            compiledActivation.supportWitnessIdentity.push_back({w.sourceContactIndex,w.pointQueryIndex,w.sourceGeometryIndex});
        }
    }
    if (preserveRestingInitialPosture && !useWeightedSkinSupport)''')
mark='            if(!restingInitialPosture.empty()) {\n                const auto posture ='
change(mark,'''            if(std::getenv("NUMI_HUMAN_PELVIC_INITIAL_ELASTIC_CSV")){
                const char* hash=std::getenv("NUMI_HUMAN_PELVIC_INITIAL_POSTURE_SHA256");
                require(hash && !restingInitialPosture.empty() && localBedLayersRequested() &&
                        loadedKneeSHA256Hex(loadedKneeFileSHA256(restingInitialPosture))==hash,
                        "regional material guess requires its matched native donor posture");
            }
'''+mark)
target.parent.mkdir(parents=True,exist_ok=True)
tmp=target.with_suffix('.tmp');tmp.write_text(s);os.replace(tmp,target)
