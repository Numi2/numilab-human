"""Run the repaired registered skin/fat in the existing native GPU scene."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("--roots", type=int, default=1)
parser.add_argument("--seconds-per-root", type=float, default=0.000015625)
parser.add_argument("--budget-wall-seconds", type=int, default=210)
parser.add_argument("--prelude-steps", type=int, default=0)
parser.add_argument("--presentation", action="store_true")
parser.add_argument("--capture-geometry-step", type=int, action="append", default=[])
parser.add_argument("--exact-contact-cache", action="store_true")
parser.add_argument("--initial-pack-verified-clone", type=Path)
parser.add_argument("--progress", action="store_true")
parser.add_argument("--parallel-constitutive-backtracking", action="store_true")
parser.add_argument("--reset-newton-fraction", action="store_true")
parser.add_argument("--tiled-deformable-limit", action="store_true")
parser.add_argument("--compensated-vascular-predictor", action="store_true")
parser.add_argument("--retained-full-volume-solver", action="store_true")
parser.add_argument("--guarded-runtime-admission", action="store_true")
parser.add_argument("--diagnostic-newton-passes", type=int, choices=(1, 2, 4))
parser.add_argument("--gravity-ramp-steps", type=int)
parser.add_argument("--vascular-pressure-ramp-steps", type=int)
parser.add_argument("--vascular-direction-witness", action="store_true")
parser.add_argument("--vascular-trace-root", type=int)
parser.add_argument("--backtracking-witness", type=int, nargs=3,
                    metavar=("ROOT", "OBJECT", "NEWTON"))
args = parser.parse_args()
assert 0 < args.seconds_per_root <= 0.002 and args.roots >= 1
steps = args.capture_geometry_step
if steps and (not args.presentation or not 0 < len(steps) <= 8 or
              len(set(steps)) != len(steps) or any(not 1 <= n <= args.roots for n in steps)):
    parser.error("geometry capture requires presentation and up to eight distinct accepted steps")
run = args.run.resolve()
run.mkdir()
roles = root/"paired-live-parent-native-role-flow"
p1 = root/"complete-fine-p1-live-parent"
mesh = root/"absolute-volume-repaired-tetgen"
sidecars = root/"paired-repaired-registration-live-parent"
compiler = root.parent/"captured-fat-sliver-495/proposal/compiler"
reg = compiler/"volume-export/registered"
known = Path.home()/"Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209"
prior_plan = known/"fine-physical-volume-706/regional-compensation-1061-root/native-import-085/install-motion.plan.json"
prior = json.loads(prior_plan.read_text())["stages"][0]
qualified_pressure_run = root.parent / "resting-clearance-and-atrium-015/motion-extension/launch.json"
pressure_runtime = root / ("guarded-runtime-admission-build"
    if args.guarded_runtime_admission else "pressure-with-abdominal-cavity-build")
pressure_metal = Path(json.loads(qualified_pressure_run.read_text())["environment"]["NUMI_MATTER_METALLIB_OVERRIDE"])
runtime = pressure_runtime / "libmetalrobo.dylib"
assert runtime.is_file() and pressure_metal.is_file()
command, env = prior["argv"], prior["environment"]
scene_build = pressure_runtime if args.guarded_runtime_admission else root/"floor-scene-build"
command[0] = str(scene_build/"numi-human-native")
command[4] = str(run/"native")
if not args.presentation:
    command.append("--mechanics-only")
command[command.index("--muscle-step-count")+1] = str(args.roots)
command[command.index("--muscle-step-seconds")+1] = repr(args.seconds_per_root)
for key in list(env):
    if (key.startswith("NUMI_H706_FRACTIONAL_RECORDED_FLOW_ADMISSION_RECEIPT_") or
        ("PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT" in key) or
        key in {"NUMI_H706_FRACTIONAL_EXACT_PROJECTION_CSV","NUMI_H706_FRACTIONAL_EXACT_PROJECTION_SHA256",
                "NUMI_H706_FRACTIONAL_SOURCE_CENTROID_CSV","NUMI_H706_FRACTIONAL_SOURCE_CENTROID_SHA256",
                "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS","NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE",
                "NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV","NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP"}):
        del env[key]
env.update(
    NUMI_H706_FRACTIONAL_RECORDED_FLOW_DIRECT_ROLE_REPLAY="1",
    NM_VASCULAR_STEP_EXTERNAL_PRESSURE="1",
    DYLD_LIBRARY_PATH=str(pressure_runtime),
    NUMI_MATTER_METALLIB_OVERRIDE=str(pressure_metal),
    NUMI_HUMAN_COUPLED_TRACE_CSV=str(run/"coupled.csv"),
    NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(run/"failure.json"),
    NUMI_HUMAN_PREPARATION_MEMORY="1",
)
if steps:
    env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = ",".join(map(str, steps))
if args.exact_contact_cache:
    env.update(
        NM_HUMAN_COMPACT_IMMUTABLE_CONTACTS="1",
        NUMI_HUMAN_CONTACT_ROW_CACHE="1",
        NUMI_HUMAN_CONTACT_AABB_EXCLUSION="1",
        NUMI_HUMAN_CONTACT_EXACT_MOTION_CACHE="1",
        NUMI_HUMAN_CONTACT_COMPACT_QUERIES="1",
    )
if args.parallel_constitutive_backtracking:
    env["NM_FEM_PARALLEL_CONSTITUTIVE_BACKTRACKING"] = "1"
if args.tiled_deformable_limit:
    env["NM_FEM_TILED_DEFORMABLE_LIMIT"] = "1"
retained_plan = None
if args.retained_full_volume_solver:
    retained_plan = root.parent/"matched-ccd-baseline-335/install-motion.plan.json"
    retained = json.loads(retained_plan.read_text())["stages"][0]["environment"]
    # Carry over the admitted full-coordinate solve. Keep current geometry,
    # reference asset IDs, physical clocks and absolute pressure ownership.
    numerical_settings = (
        "NM_CONTACT_COMPLETED_GEOMETRY_CACHE", "NM_CONTACT_COMPLETED_INCIDENCE_REUSE",
        "NM_CONTACT_FINAL_ZERO_GEOMETRY_REUSE", "NM_FEM_CACHED_NODE_SIMD",
        "NM_FEM_CLEARANCE_ALIGNED_BARRIER", "NM_FEM_COMPENSATED_CONTACT_PLANE_CCD",
        "NM_FEM_FRESH_DIRECTION_ALPHA_RESET", "NM_FEM_FRESH_RHS_PRECONDITIONER_PREFLIGHT",
        "NM_FEM_GATHER_SIMD", "NM_FEM_HUMAN_PARTITIONED_FUSED_TANGENT",
        "NM_FEM_HUMAN_PARTITIONED_SPARSE_BASIS", "NM_FEM_NEWTON_TAIL_ICB",
        "NM_FEM_PARALLEL_CONSTITUTIVE_BACKTRACKING", "NM_FEM_REFERENCE_SELF_CONTACT_WIDTH",
        "NM_FEM_RESOLVED_MOLLIFIER_CURVATURE", "NM_FEM_SHARE_IMMUTABLE_TETRAHEDRA",
        "NM_FEM_TANGENT_OWNER_TASKS", "NM_FEM_TILED_DEFORMABLE_LIMIT",
        "NM_FGMRES_AGGREGATE_CYCLE_ICB", "NM_FGMRES_AGGREGATE_FIRST_CYCLE_ICB",
        "NM_FGMRES_COUPLED_BORROWED_ENCODER", "NM_FGMRES_FLAT_COLUMN_ICB",
        "NM_FGMRES_INEXACT_COUPLED_LINEAR_CERTIFICATE", "NM_FGMRES_INEXACT_NEWTON_FORCING",
        "NM_FGMRES_INEXACT_NEWTON_MAX_ETA", "NM_FGMRES_INEXACT_NEWTON_TIGHTENING",
        "NM_FGMRES_LEFT_ROW_EQUILIBRATION_COUPLED_FEM", "NM_FGMRES_PARALLEL_BASE_PRECONDITIONER",
        "NM_FGMRES_PARALLEL_BASE_PRECONDITIONER_MASK", "NM_FGMRES_RELATIVE_INNER_FORCING",
        "NM_FGMRES_SKIP_COMPLETED_CYCLE_WORK", "NM_FGMRES_SKIP_COMPLETED_LINEAR_BUILDERS",
        "NM_FGMRES_SKIP_COMPLETED_ZERO_OPERATOR_ACTION", "NM_MATTER_CONTACT_ZERO_STEP_GEOMETRY_REUSE",
        "NM_MIXED_CERTIFICATE_OWNER_INCIDENCE", "NM_MIXED_CERTIFICATE_TILED",
        "NM_SCHEDULER_TILED_FEM_OBSERVE", "NM_VASCULAR_COMPENSATED_CAVITY_WORDS",
        "NM_VASCULAR_COOPERATIVE_CAVITY_OPERATOR", "NM_VASCULAR_PERSISTENT_LOW_WORDS",
        "NUMI_MATTER_HUMAN_PARTITIONED_FEM_TANGENT", "NUMI_MATTER_HUMAN_PARTITIONED_PRECONDITIONER",
        "NUMI_MATTER_PARALLEL_WORKING_SET_CLEAR", "NUMI_MATTER_SPARSE_TOPOLOGY_INCIDENCE",
        "NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS",
    )
    env.update((key, retained[key]) for key in numerical_settings)
if args.compensated_vascular_predictor:
    env.update(
        NM_VASCULAR_PERSISTENT_LOW_WORDS="1",
        NM_VASCULAR_COMPENSATED_CAVITY_WORDS="1",
        NUMI_MATTER_VASCULAR_BLOCK_PREDICTOR="1",
        NM_FGMRES_PREFIX_PRECONDITIONER_ICB="0",
        NM_FGMRES_FLAT_COLUMN_ICB="0",
        NM_FGMRES_AGGREGATE_CYCLE_ICB="0",
        NM_FGMRES_AGGREGATE_FIRST_CYCLE_ICB="0",
    )
if args.reset_newton_fraction:
    env["NM_FEM_FRESH_DIRECTION_ALPHA_RESET"] = "1"
if args.progress:
    env.update(NUMI_GPU_PROGRESS="1", NUMI_GPU_PROGRESS_ROOT="0")
if args.diagnostic_newton_passes is not None:
    env.update(
        NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS=str(args.diagnostic_newton_passes),
        NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT="0",
        NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA="1",
        NM_VASCULAR_TRACE_ROOT="0",
    )
if args.gravity_ramp_steps is not None:
    assert 2 <= args.gravity_ramp_steps <= args.roots + args.prelude_steps
    command += ["--resting-gravity-ramp-steps", str(args.gravity_ramp_steps)]
if args.vascular_pressure_ramp_steps is not None:
    assert 2 <= args.vascular_pressure_ramp_steps <= args.roots + args.prelude_steps - 7
    command += ["--resting-vascular-pressure-ramp-steps", str(args.vascular_pressure_ramp_steps)]
if args.vascular_trace_root is not None:
    if not 0 <= args.vascular_trace_root < args.roots:
        parser.error("selected vascular trace root is outside the requested horizon")
    env["NM_VASCULAR_TRACE_ROOT"] = str(args.vascular_trace_root)
if args.backtracking_witness is not None:
    witness_root, witness_object, witness_newton = args.backtracking_witness
    if not (0 <= witness_root < args.roots and witness_object >= 0 and
            0 <= witness_newton < int(env["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"])):
        parser.error("backtracking witness indices are outside the configured horizon")
    if env.get("NM_FEM_PARALLEL_CONSTITUTIVE_BACKTRACKING") != "1":
        parser.error("the bounded witness requires the tiled constitutive check")
    env.update(
        NM_FEM_BACKTRACKING_TRACE_ROOT=str(witness_root),
        NM_FEM_BACKTRACKING_TRACE_OBJECT=str(witness_object),
        NM_FEM_BACKTRACKING_TRACE_NEWTON=str(witness_newton),
    )
if args.vascular_direction_witness:
    env["NUMI_MATTER_VASCULAR_ROW16_DIRECTION_TRACE"] = "1"
if args.seconds_per_root > .000015625:
    exponent = round(__import__("math").log2(args.seconds_per_root/.000015625))
    assert args.seconds_per_root == .000015625*2**exponent
    env["NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT"]=str(exponent)
if args.prelude_steps:
    assert 8 <= args.prelude_steps <= 4096
    env["NUMI_HUMAN_SUPPORTED_PRELUDE_STEPS"] = str(args.prelude_steps)
    env["NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT"] = "0"
    env["NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT"] = "0"

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

if args.initial_pack_verified_clone:
    initial = args.initial_pack_verified_clone.resolve(strict=True)
    env["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"] = str(initial)
    env["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"] = digest(initial)

def put(path_key, hash_key, path, verified_hash=None):
    env[path_key] = str(path)
    env[hash_key] = verified_hash or digest(path)

prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
def asset(path_key, hash_key, path, verified_hash=None):
    put(prefix+path_key,prefix+hash_key,path,verified_hash)

packet = root/"floor-live-parent-capture-002/compiled-parent-cell-mass.csv"
asset("PARENT_PACKET_CSV","PARENT_PACKET_SHA256",packet)
asset("PARENT_LABELS_CSV","PARENT_LABELS_SHA256",Path(str(packet)+".labels.csv"))
for kind in ("NODES","SOURCE","TETS"):
    env[prefix+"PARENT_"+kind+"_SHA256"] = digest(Path(str(packet)+"."+kind.lower()+".csv"))
asset("TETGEN_NODE","TETGEN_NODE_SHA256",mesh/"mesh.node")
asset("TETGEN_ELEMENTS","TETGEN_ELEMENTS_SHA256",mesh/"mesh.ele")
asset("CHILD_LABELS","CHILD_LABELS_SHA256",root/"paired-floor-child.labels.csv")
for kind, path in {
    "COMBINED_NODE_MAP":sidecars/"combined-node-map.csv",
    "FACET_ANCESTRY":sidecars/"facet-ancestry-current.csv",
    "DEEP_ATTACHMENT_REQUESTS":compiler/"bindings/deep-attachment-requests.csv",
    "DONOR_FRAMES":reg/"donor-frames.csv",
    "NODE_REGISTRATION":sidecars/"node-parent-registration.csv",
    "OUTER_FACE_ANCESTRY":reg/"outer-face-ancestry.csv",
}.items():
    asset(kind+"_CSV",kind+"_SHA256",path)
env[prefix+"MAX_PARENT_REGISTRATION_DISTANCE_M"] = "0.02"
pins = [line.split("\t") for line in (sidecars/"source-domain-pins.tsv").read_text().splitlines()]
for tag,(domain,sha,vertices,faces) in zip("OID",pins):
    role = f"ROLE_{tag}_"
    asset(f"SURFACE_{tag}_NODES_CSV",f"SURFACE_{tag}_NODES_SHA256",sidecars/f"registered-{tag}.csv")
    asset(f"SURFACE_{tag}_FACES_CSV",f"SURFACE_{tag}_FACES_SHA256",reg/f"registered-{tag}-faces.csv")
    weights=compiler/"volume-export/roles"/f"{tag}-full86.csv"
    proof=json.loads(Path(str(weights)+".zst.recovery.json").read_text())
    if not weights.exists():
        temporary=weights.with_suffix(".csv.incomplete")
        subprocess.run(["/opt/homebrew/bin/zstd","-qd",str(weights)+".zst","-o",temporary],check=True)
        assert digest(temporary)==proof["sha256"]
        temporary.rename(weights)
    asset(role+"FULL86_CSV",role+"FULL86_SHA256",weights,proof["sha256"])
    asset(role+"CONTACT_THICKNESS_CSV",role+"CONTACT_THICKNESS_SHA256",
          compiler/"volume-export/roles"/f"{tag}-contact-thickness.csv")
    source=compiler/"bindings"/f"{tag}-source.json"
    binding_receipt=json.loads((compiler/"bindings/receipt.json").read_text())
    assert binding_receipt["inputs"][str(compiler/"volume-export/roles"/source.name)]==sha
    binding_source_sha=binding_receipt["outputs"][str(source)]["sha256"]
    assert digest(source)==binding_source_sha
    asset(role+"SOURCE_ASSET",role+"SOURCE_ASSET_SHA256",source,binding_source_sha)
    env[prefix+role+"SOURCE_DOMAIN"]=domain
    env[prefix+role+"SOURCE_VERTEX_COUNT"]=vertices
    env[prefix+role+"SOURCE_FACE_COUNT"]=faces
    if tag!="O":
        env[prefix+("D37_INNER_FACE_COUNT" if tag=="I" else "D37_DEEP_FACE_COUNT")]=faces

fractional="NUMI_H706_FRACTIONAL_"
env[fractional+"STATE_SHA256"]=digest(p1/"current-paired-volume-state.json")
env[fractional+"SOURCE330_MASS_SHA256"]=digest(packet)
for key in ("NODES","TETS"):
    put(fractional+"SOURCE330_"+key,fractional+"SOURCE330_"+key+"_SHA256",Path(str(packet)+"."+key.lower()+".csv"))
witnesses=json.loads((root/"paired-live-parent-flow-witness/receipt.json").read_text())
stems=["CHILD_MASS","TRANSPORT_FLOW","SOURCE_CARRIER_FLOW","CHILD_SOURCE_DONOR","SOURCE_PARENT","SOURCE_PARENT_DONOR"]
for name, witness in zip(("skin","fat"),witnesses["roles"]):
    key=name.upper()
    support=json.loads((p1/f"{name}-p1-receipt.json").read_text())
    put(fractional+f"P1_{key}_NPZ",fractional+f"P1_{key}_SHA256",Path(support["p1_one_hop_path"]),support["p1_one_hop_sha256"])
    put(fractional+f"CARRIERS_{key}_NPZ",fractional+f"CARRIERS_{key}_SHA256",Path(witness["carrier_path"]),witness["carrier_sha256"])
    put(fractional+f"CARRIER_WITNESS_{key}_CSV",fractional+f"CARRIER_WITNESS_{key}_SHA256",Path(witness["witness_path"]),witness["witness_sha256"])
    env[fractional+f"GEOMETRY_{key}_SHA256"]=env[fractional+f"VOLUME_{key}_SHA256"]=support["role_geometry_sha256"]
    recorded="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_"+key+"_"
    receipt=roles/f"{name}-recorded-flow-receipt.json"
    put(recorded+"RECEIPT_JSON",recorded+"RECEIPT_SHA256",receipt)
    hashes=json.loads(receipt.read_text())["flow_csv"]
    for stem in stems:
        name_on_disk=name+"-"+stem.lower().replace("_","-")+".csv"
        put(recorded+stem+"_CSV",recorded+stem+"_CSV_SHA256",roles/name_on_disk,hashes[name_on_disk]["sha256"])
pins={command[0]:digest(Path(command[0])),str(prior_plan):digest(prior_plan),
      str(runtime):digest(runtime),str(pressure_metal):digest(pressure_metal),
      str(qualified_pressure_run):digest(qualified_pressure_run)}
if retained_plan is not None:
    pins[str(retained_plan)] = digest(retained_plan)
(run/"launch.json").write_text(json.dumps(dict(argv=command,environment=env,cwd=prior["cwd"],pins=pins),indent=2)+"\n")
environment={key:value for key,value in os.environ.items() if not key.startswith(("NM_","NUMI_"))}
environment.update(env)
start=time.monotonic()
with (run/"stdout.log").open("wb") as out, (run/"stderr.log").open("wb") as err:
    with subprocess.Popen(command,env=environment,cwd=prior["cwd"],stdout=out,stderr=err) as proc:
        try:
            proc.wait(timeout=args.budget_wall_seconds)
            stop="native_exit"
        except subprocess.TimeoutExpired:
            stop="bounded_wall_budget_expired"
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
execution={"exit":proc.returncode,"stop":stop,"wall_seconds":time.monotonic()-start}
(run/"execution.json").write_text(json.dumps(execution,indent=2)+"\n")
print(json.dumps(execution))
