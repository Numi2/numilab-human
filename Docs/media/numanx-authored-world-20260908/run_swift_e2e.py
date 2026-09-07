from pathlib import Path
import json, os, signal, subprocess, time
base = Path('/Users/n/human-completion-20260907')
root = Path('/Users/n/MetalRobo-human-completion-20260907')
build = Path('/Users/n/MetalRobo-human-completion-build-20260907')
brain = Path('/Users/n/numi-brain-human-completion-20260907')
fixture = base/'authored-world-fixture'
identity = json.loads((fixture/'authored-100us.json').read_text())
selected = {
 'NUMANX_METALROBO_LIBRARY': str(build/'lib/libmetalrobo.dylib'),
 'NUMANX_FULLBODY_RIGID': str(base/'input/myosim-fullbody-core-reference.nhrigid'),
 'NUMANX_FULLBODY_MUSCLE': str(base/'input/myosim-fullbody-muscle-reference.nhmyo'),
 'NUMANX_FULLBODY_CONTACT': str(base/'input/myosim-fullbody-support-contact.nhcnt'),
 'NUMANX_METALROBO_METALLIB': str(build/'shaders/MetalRobo.metallib'),
 'NUMANX_MATTER_METALLIB': str(build/'matter/shaders/NumiMatter.metallib'),
 'NUMANX_FULLBODY_VISUAL_PACK': '/Users/n/NumiHumanCurrent/anterior-thorax-smoke/myosim-fullbody-articulated-markers-muscle-driven-selected-actuators-source-support-contact.mrvpack',
 'NUMANX_FULLBODY_VISION_PROFILE': '/Users/n/Numan-X-neuron-v1/assets/numanx-head-vision-profile.v1.json',
 'NUMANX_MATTER_MATERIAL': str(root/'matter/materials/silicone.nmatter'),
 'NUMANX_MATTER_WORLD_PACKAGE': str(fixture/'authored-100us.nmatterpack'),
 'NUMANX_HUMAN_SOURCE_FP': identity['human_source_fp'],
 'NUMANX_MATTER_WORLD_FP': identity['matter_world_fp'],
}
environment = dict(os.environ, **selected)
environment['PATH'] = '/opt/homebrew/bin:' + environment.get('PATH','')
command = ['swift','test','--skip-build','--filter',
 'testRealFullBodyBrainProposalApplyAndJointPublication|testAuthoredMatterBrainProposalApplyAndJointPublication']
started = time.monotonic()
with (base/'authored-world-swift-e2e-final.log').open('w') as output:
 process = subprocess.Popen(command, cwd=brain, env=environment, start_new_session=True,
                            stdout=output, stderr=subprocess.STDOUT)
 try:
  returncode = process.wait(timeout=60)
 except subprocess.TimeoutExpired:
  os.killpg(process.pid, signal.SIGTERM)
  try: process.wait(timeout=5)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGKILL); process.wait()
  returncode = 124
(base/'authored-world-swift-e2e-launch.json').write_text(json.dumps({
 'command':command, 'cwd':str(brain), 'environment':selected,
 'elapsed_seconds':time.monotonic()-started, 'returncode':returncode}, indent=2)+'\n')
raise SystemExit(returncode)
