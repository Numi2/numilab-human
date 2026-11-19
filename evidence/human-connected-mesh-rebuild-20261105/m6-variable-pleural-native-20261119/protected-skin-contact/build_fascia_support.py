"""Align the shared scene with the physical weighted skin before reference tissue cooking."""
import difflib
import os
import subprocess

from run_support_dependency_preflight import HERE, read, write, digest

base=HERE/"shared-fascia-pressure-build"
build=HERE/"shared-fascia-support-build"
build.mkdir()
unit="numilab_human_myosim_visual_probe.mm"
app="numi-human-native"
for p in (*base.glob("*.hpp"),base/"runtime.mm.o"):
    os.link(p,build/p.name)
(build/"metalrobo").mkdir()
for p in (base/"metalrobo").iterdir():
    os.link(p,build/"metalrobo"/p.name)
original=(base/unit).read_text()
start="""                } else {
                    aligned=makeGroundAlignedSupport(rigid.model,*supportContactPayload,poseQ);
                }
                require(aligned.q.size()"""
replacement="""                } else {
                    aligned=makeGroundAlignedSupport(rigid.model,*supportContactPayload,poseQ);
                    const char* sharedFascia=std::getenv("NUMI_HUMAN_RESTING_PASSIVE_FASCIA");
                    if (sharedFascia && std::string_view(sharedFascia)=="1" && skinPayload) {
                        // Register the tissue reference in the same first-touch
                        // placement as the detailed skin used by the weighted
                        // bed contact. The legacy support primitives are only
                        // a seed and can start the actual surface below the bed.
                        const auto seed=packMetalConfiguration(aligned.q);
                        metalrobo::MetalArticulatedOperatorInput input;
                        input.articulationIndex=0u;
                        input.environmentCount=1u;
                        input.q=seed;
                        metalrobo::MetalArticulatedOperatorConfig config;
                        config.pointJacobiansOnly=true;
                        metalrobo::MetalArticulatedOperatorResult result;
                        const auto status=metalrobo::runMetalArticulatedOperator(rigid.model,input,result,config);
                        require(status.succeeded() && status.dispatched &&
                                    result.bodyPoses.size()==rigid.model.bodies.size(),
                                "shared fascia first-touch body frames are invalid");
                        const auto states=visualBodyStates(rigid.model,result.bodyPoses);
                        const auto& h=supportContactPayload->header;
                        const auto n=normalizedVector({h.groundNormalX,h.groundNormalY,h.groundNormalZ},
                                                       "shared fascia bed normal");
                        std::vector<bool> used(skinPayload->vertices.size(),false);
                        for (const auto i:skinPayload->indices) used.at(i)=true;
                        double minGap=std::numeric_limits<double>::infinity();
                        for (std::size_t i=0;i<used.size();++i) {
                            if (!used[i]) continue;
                            const auto p=skinVertexWorld(*skinPayload,skinPayload->vertices[i],i,states);
                            minGap=std::min(minGap,n[0]*(double(p.x)-h.groundPointX)+
                                                   n[1]*(double(p.y)-h.groundPointY)+
                                                   n[2]*(double(p.z)-h.groundPointZ));
                        }
                        require(std::isfinite(minGap),"shared fascia first-touch surface has no valid referenced vertex");
                        // Round outward by a Float32-scale margin before
                        // handing off to the independent weighted GPU witness.
                        const double correction=std::max(0.0,8e-6-minGap);
                        for (std::size_t i=0;i<3u;++i) aligned.q[articulation.qOffset+i]+=correction*n[i];
                        std::cout<<std::setprecision(17)<<"resting_shared_fascia_skin_first_touch"
                                 <<" source_gap_m="<<minGap<<" outward_shift_m="<<correction
                                 <<" force_rows_added=0 scope=fresh_reference_initialization"<<std::endl;
                    }
                }
                require(aligned.q.size()"""
assert original.count(start)==1
updated=original.replace(start,replacement)
(build/unit).write_text(updated)
(HERE/"shared-fascia-initial-support.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True),updated.splitlines(keepends=True),
    "a/apps/"+unit,"b/apps/"+unit)))
paths={str(base/f):str(build/f) for f in (app,app+".o",unit,"runtime.mm.o")}
commands=[[paths.get(a,a) for a in cmd] for cmd in read(base/"build-commands.json")[-2:]]
write(build/"build-commands.json",commands)
for cmd in commands:
    subprocess.run(cmd,cwd="/Users/numi/Work/skin-fat-native-integration-209/build-native",check=True)
print(digest(build/app),flush=True)
