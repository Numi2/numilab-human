"""Build the source-scoped abdominal fascia mass transfer against the existing owners."""
import difflib
import os
import subprocess

from run_support_dependency_preflight import HERE, read, write, digest

base=HERE/"recomputed-support-build"
build=HERE/"shared-fascia-mass-build"
build.mkdir()
work=read(HERE/"recomputed-support-preflight/launch.json")
unit="numilab_human_myosim_visual_probe.mm"
app="numi-human-native"
original=(base/unit).read_text()
updated=original
def replace(a,b):
    global updated
    assert updated.count(a)==1,a
    updated=updated.replace(a,b)
replace("""    std::span<const std::uint32_t> sourceBodyMap,
    bool includeInferredACL,""","""    std::span<const std::uint32_t> sourceBodyMap,
    const metalrobo::NumiHumanLoadedKneeDigest& rigidPayloadSHA,
    bool includeInferredACL,""")
replace("""                        activeSeparationBodyMap,
                        selectInferredACL&&bonePayload.has_value()&&""","""                        activeSeparationBodyMap,rigid.payloadSha256,
                        selectInferredACL&&bonePayload.has_value()&&""")
replace("""    double fasciaCookedMassKg=0.0;""","""    double fasciaCookedMassKg=0.0;
    std::vector<metalrobo::NumiHumanTissueMassContribution> abdominalSourceMass;
    // The registered abdomen is an attachment frame with only 10 g in the
    // source rig. For its explicit oblique sheets, exchange mass with the
    // composite trunk budget at the torso frame. Preserve each source
    // contribution's world position so the existing partition verifies
    // mass, first moments and inertia. The attachment and contact frames
    // remain on the authored route.
    constexpr std::uint32_t kReferenceAbdomen=7u, kReferenceTorso=20u;
    constexpr std::string_view kFasciaTrunkReferenceRigidSHA=
        "c10924a45efc1cfb3916bd22baabeeca491da5e412bbaad37b1328fe364db53d";
    const bool useFasciaTrunkMassReference =
        sourceBodyMap.empty() && loadedKneeSHA256Hex(rigidPayloadSHA)==kFasciaTrunkReferenceRigidSHA;""")
replace("""            fasciaCookedMassKg+=double(cookedMass);
            massNodes.push_back({globalNode,donorBody,double(cookedMass),
                {donorLocal.x,donorLocal.y,donorLocal.z}});""","""            fasciaCookedMassKg+=double(cookedMass);
            massNodes.push_back({globalNode,donorBody,double(cookedMass),
                {donorLocal.x,donorLocal.y,donorLocal.z}});
            if(useFasciaTrunkMassReference &&
               regionIndex>=kLatissimusFasciaRegionEnd && donorBody==kReferenceAbdomen) {
                require(kReferenceTorso<bodies.size() &&
                            (region.softTissueStableId&~kFasciaRouteBodyBit)==kReferenceAbdomen,
                        "abdominal fascia trunk source frame identity drifted");
                const auto& torso=bodies[kReferenceTorso];
                const auto& q=torso.orientation;
                const mr_float4 local=rotatePoint(mr_float4{-q.x,-q.y,-q.z,q.w},
                    femSubtract(points[sourceGlobal],torso.position));
                const auto reconstructed=femAdd(torso.position,rotatePoint(q,local));
                require(femLength(femSubtract(reconstructed,points[sourceGlobal]))<=1e-6f,
                        "abdominal fascia source-mass transformation moved its reference point");
                abdominalSourceMass.push_back({globalNode,kReferenceTorso,double(cookedMass),
                    {local.x,local.y,local.z}});
            }""")
replace("""    std::vector<metalrobo::NumiHumanTissueMassContribution> massContributions;
    const auto appendContributions""","""    if(!abdominalSourceMass.empty()) {
        double transferKg=0.0;
        for(const auto& row:abdominalSourceMass)transferKg+=row.massKg;
        std::cout<<"resting_abdominal_fascia_source_mass owner=composite_reference_torso"
                 <<" anchor_body="<<kReferenceAbdomen<<" source_body="<<kReferenceTorso
                 <<" nodes="<<abdominalSourceMass.size()<<" mass_kg="<<transferKg
                 <<" rigid_sha256="<<kFasciaTrunkReferenceRigidSHA
                 <<" total_mass_added_kg=0 interpretation=inferred_trunk_mass_partition"
                 <<std::endl;
    }
    auto massContributions=std::move(abdominalSourceMass);
    const auto appendContributions""")
(build/unit).write_text(updated)
patch="".join(difflib.unified_diff(original.splitlines(keepends=True),updated.splitlines(keepends=True),
                                    "a/apps/"+unit,"b/apps/"+unit))
(HERE/"shared-fascia-abdominal-mass.patch").write_text(patch)
for p in base.glob("*.hpp"):
    os.link(p,build/p.name)
(build/"metalrobo").mkdir()
for p in (base/"metalrobo").iterdir():
    os.link(p,build/"metalrobo"/p.name)
paths={str(base/f):str(build/f) for f in (app,app+".o",unit)}
cmds=[[paths.get(a,a) for a in cmd] for cmd in read(base/"build-commands.json")]
write(build/"build-commands.json",cmds)
for cmd in cmds:
    subprocess.run(cmd,cwd="/Users/numi/Work/skin-fat-native-integration-209/build-native",check=True)
print(digest(build/app),flush=True)
