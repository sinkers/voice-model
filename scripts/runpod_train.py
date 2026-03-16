#!/usr/bin/env python3
"""
RunPod Piper Training Pipeline — Andrew Voice Model
Reliable re-runnable script. Run from the voice-model project directory.

Usage:
    python3 scripts/runpod_train.py                   # full run
    python3 scripts/runpod_train.py --monitor POD_ID  # resume monitoring
    python3 scripts/runpod_train.py --download POD_ID # just download finished model

All state saved to output/runpod_state.json for resumability.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RUNPOD_API_KEY  = os.environ.get("RUNPOD_API_KEY", "") or open(os.path.expanduser("~/.runpod_api_key")).read().strip()
DATASET_URL  = "https://litter.catbox.moe/t7ix4t.zip"
GPU_TYPE        = "NVIDIA RTX A5000"
# PyTorch 2.1 + Python 3.10: PL 1.7.7 has wheels for py3.10 and works with torch 2.x
DOCKER_IMAGE    = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
EPOCHS          = 6000
PROJECT_DIR     = Path(__file__).parent.parent
OUTPUT_DIR      = PROJECT_DIR / "output"
SSH_KEY_PATH    = Path.home() / ".ssh" / "runpod_piper"
GRAPHQL_URL     = "https://api.runpod.io/graphql"
# ──────────────────────────────────────────────────────────────────────────────

# Full training script that runs on the pod — all fixes baked in
TRAIN_SCRIPT = r"""#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/train.log) 2>&1

echo "========================================"
echo " Piper Training Pipeline — Andrew Voice"
echo " Started: $(date)"
echo "========================================"

# ── 1. Install system deps ────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -q espeak-ng espeak-ng-data libespeak-ng-dev
echo "  espeak-ng OK"

# ── 2. Download and extract dataset ──────────────────────────────────────────
echo "[2/6] Downloading dataset..."
mkdir -p /workspace/ljspeech/wavs /workspace/output

if [ ! -f /tmp/dataset.zip ]; then
  curl -L -o /tmp/dataset.zip "$DATASET_URL"
fi

python3 -c "
import zipfile, os
z = zipfile.ZipFile('/tmp/dataset.zip')
for name in z.namelist():
    if name.endswith('.wav'):
        with z.open(name) as src, open(f'/workspace/ljspeech/wavs/{os.path.basename(name)}', 'wb') as dst:
            dst.write(src.read())
    elif name == 'metadata.csv':
        with z.open(name) as src, open('/workspace/ljspeech/metadata.csv', 'wb') as dst:
            dst.write(src.read())
print(f'  WAVs: {len(os.listdir(\"/workspace/ljspeech/wavs\"))}')
print(f'  Metadata: {len(open(\"/workspace/ljspeech/metadata.csv\").readlines())} lines')
"

# ── 3. Install piper-train (PyTorch 1.13 = exact match) ──────────────────────
echo "[3/6] Installing piper-train..."
pip install -q "cython<3"
# Install PL 1.7.7 with --no-deps first to bypass torch<2 constraint check, then add deps
pip install -q --no-deps pytorch-lightning==1.7.7
pip install -q torchmetrics==0.11.4 piper-phonemize librosa onnxruntime fsspec packaging
pip install -q "pyDeprecate>=0.3.1" "PyYAML>=5.4" "tqdm>=4.57.0" "tensorboard>=2.9.1"

# Get piper source
curl -L -o /tmp/piper.zip 'https://github.com/rhasspy/piper/archive/refs/heads/master.zip'
python3 -c "import zipfile; zipfile.ZipFile('/tmp/piper.zip').extractall('/workspace/')"
cd /workspace/piper-master/src/python
pip install -q --no-deps -e .

# Build Cython extension (required)
echo "  Building monotonic_align..."
bash build_monotonic_align.sh 2>/dev/null
python3 -c "from piper_train.vits.monotonic_align import maximum_path; print('  monotonic_align OK')"

# Patch lightning.py for torch 2.x compatibility (PL 1.7.7 validator broken)
echo "  Patching lightning.py for torch 2.x..."
python3 -c "
import re, sys
f = '/workspace/piper-master/src/python/piper_train/vits/lightning.py'
c = open(f).read()
# Return scheduler dicts instead of raw schedulers
c = c.replace(
    '        return optimizers, schedulers',
    '''        scheduler_configs = [
            {\"scheduler\": schedulers[0], \"interval\": \"epoch\", \"frequency\": 1},
            {\"scheduler\": schedulers[1], \"interval\": \"epoch\", \"frequency\": 1},
        ]
        return optimizers, scheduler_configs'''
)
# Add lr_scheduler_step to bypass PL validator
c = c.replace(
    '    @staticmethod\n    def add_model_specific_args(parent_parser):',
    '''    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    @staticmethod
    def add_model_specific_args(parent_parser):'''
)
open(f, 'w').write(c)
print('  lightning.py patched OK')
"

# ── 4. Preprocess ─────────────────────────────────────────────────────────────
echo "[4/6] Preprocessing dataset..."
rm -rf /workspace/training
python3 -m piper_train.preprocess \
  --language en \
  --input-dir /workspace/ljspeech \
  --output-dir /workspace/training \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050 \
  --max-workers 4
echo "  Preprocessing complete"

# ── 5. Train ──────────────────────────────────────────────────────────────────
echo "[5/6] Training $EPOCHS epochs..."
RESUME_ARG=""
if ls /workspace/training/lightning_logs/version_0/checkpoints/epoch*.ckpt 2>/dev/null | grep -q .; then
    LAST=$(ls /workspace/training/lightning_logs/version_0/checkpoints/epoch*.ckpt | sort -V | tail -1)
    RESUME_ARG="--resume_from_checkpoint $LAST"
    echo "  Resuming from: $LAST"
fi

python3 -m piper_train \
  --dataset-dir /workspace/training \
  --accelerator gpu \
  --devices 1 \
  --batch-size 32 \
  --validation-split 0.0 \
  --num-test-examples 0 \
  --max_epochs $EPOCHS \
  --checkpoint-epochs 100 \
  --precision 32 \
  $RESUME_ARG

# ── 6. Export ONNX ────────────────────────────────────────────────────────────
echo "[6/6] Exporting ONNX model..."
CKPT=$(ls /workspace/training/lightning_logs/version_0/checkpoints/epoch*.ckpt | sort -V | tail -1)
echo "  Checkpoint: $CKPT"

python3 -m piper_train.export_onnx "$CKPT" /workspace/output/andrew-medium.onnx
cp /workspace/training/config.json /workspace/output/andrew-medium.onnx.json

# Patch config to piper inference format
python3 -c "
import json
with open('/workspace/output/andrew-medium.onnx.json') as f:
    cfg = json.load(f)
cfg['audio']['quality'] = 'medium'
cfg['espeak']['voice'] = 'en'
with open('/workspace/output/andrew-medium.onnx.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('  Config OK')
"

echo ""
echo "========================================"
echo " TRAINING COMPLETE: $(date)"
echo " $(ls -lh /workspace/output/)"
echo "========================================"
echo "DONE" > /workspace/training_status.txt
echo "Waiting for Pi to download model before pod terminates..."
# Pod stays alive — the monitor script on the Pi will download then terminate
"""

TRAIN_SCRIPT = TRAIN_SCRIPT \
    .replace("$DATASET_URL", DATASET_URL) \
    .replace("$EPOCHS", str(EPOCHS)) \
    .replace("$RUNPOD_API_KEY", RUNPOD_API_KEY)


def graphql(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    r = subprocess.run(
        ["curl", "-s", "--max-time", "30",
         "-H", "Content-Type: application/json",
         "-H", f"Authorization: Bearer {RUNPOD_API_KEY}",
         "-d", json.dumps(payload),
         GRAPHQL_URL],
        capture_output=True, text=True
    )
    data = json.loads(r.stdout)
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


def ensure_ssh_key():
    SSH_KEY_PATH.parent.mkdir(exist_ok=True)
    if not SSH_KEY_PATH.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(SSH_KEY_PATH), "-N", "", "-C", "runpod-piper"],
            check=True, capture_output=True
        )
    return (SSH_KEY_PATH.parent / (SSH_KEY_PATH.name + ".pub")).read_text().strip()


def ensure_network_volume():
    """Create or reuse a persistent network volume for training data."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    state_file = OUTPUT_DIR / "runpod_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    if state.get("network_volume_id"):
        print(f"  Using existing network volume: {state['network_volume_id']}")
        return state["network_volume_id"]

    # Find existing volume by name first
    query = '{ myself { networkVolumes { id name size } } }'
    vols = graphql(query)["myself"]["networkVolumes"]
    for v in vols:
        if v["name"] == "piper-training":
            print(f"  Found existing network volume: {v['id']}")
            state["network_volume_id"] = v["id"]
            state_file.write_text(json.dumps(state, indent=2))
            return v["id"]

    # Create new 20GB network volume
    create_query = """
    mutation { createNetworkVolume(input: {
        name: "piper-training"
        size: 20
        dataCenterId: "EU-RO-1"
    }) { id name size } }
    """
    vol = graphql(create_query)["createNetworkVolume"]
    print(f"  Created network volume: {vol['id']} ({vol['size']}GB)")
    state["network_volume_id"] = vol["id"]
    state_file.write_text(json.dumps(state, indent=2))
    return vol["id"]


def create_pod(pub_key):
    print("[2a/4] Ensuring persistent network volume...")
    network_volume_id = ensure_network_volume()

    query = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) { id }
    }
    """
    variables = {"input": {
        "cloudType": "SECURE",
        "gpuCount": 1,
        "containerDiskInGb": 40,
        "minVcpuCount": 4,
        "minMemoryInGb": 15,
        "gpuTypeId": GPU_TYPE,
        "name": "andrew-piper-training",
        "imageName": DOCKER_IMAGE,
        "ports": "22/tcp",
        "networkVolumeId": network_volume_id,
        "startJupyter": False,
        "startSsh": True,
        "env": [{"key": "PUBLIC_KEY", "value": pub_key}],
    }}
    return graphql(query, variables)["podFindAndDeployOnDemand"]["id"]


def get_ssh_info(pod_id):
    query = """query { pod(input:{podId:$id}) { runtime { ports { ip privatePort publicPort } } } }"""
    query = query.replace("$id", f'"{pod_id}"')
    pod = graphql(query)["pod"]
    if not pod or not pod.get("runtime"):
        return None
    ports = pod["runtime"].get("ports") or []
    return next((p for p in ports if p["privatePort"] == 22), None)


def ssh(ssh_info, cmd, check=True, capture=True):
    args = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=8",
        "-o", "BatchMode=yes", "-o", "ServerAliveInterval=10",
        "-i", str(SSH_KEY_PATH),
        "-p", str(ssh_info["publicPort"]),
        f"root@{ssh_info['ip']}", cmd
    ]
    r = subprocess.run(args, capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"SSH failed: {r.stderr[:200]}")
    return r.stdout.strip()


def wait_for_ssh(pod_id, timeout=400):
    print("  Waiting for pod", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = get_ssh_info(pod_id)
        if info:
            r = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=6",
                 "-o", "BatchMode=yes", "-i", str(SSH_KEY_PATH),
                 "-p", str(info["publicPort"]), f"root@{info['ip']}", "echo ok"],
                capture_output=True, text=True
            )
            if "ok" in r.stdout:
                print(" ready!")
                return info
        print(".", end="", flush=True)
        time.sleep(12)
    raise TimeoutError("SSH never became available")


def scp_get(ssh_info, remote, local):
    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no", "-i", str(SSH_KEY_PATH),
         "-P", str(ssh_info["publicPort"]),
         f"root@{ssh_info['ip']}:{remote}", str(local)],
        check=True
    )


def terminate_pod(pod_id):
    graphql('mutation { podTerminate(input: { podId: "' + pod_id + '" }) }')


def monitor(pod_id, ssh_info):
    print("  Monitoring training (Ctrl+C to detach — pod keeps running)")
    print()
    while True:
        try:
            status = ssh(ssh_info, "cat /workspace/training_status.txt 2>/dev/null || echo RUNNING", check=False)
            if status.strip() == "DONE":
                print("\n  ✅ Training complete!")
                return True

            # Get progress
            epoch = ssh(ssh_info, "grep -oP 'Epoch \\K[0-9]+/[0-9]+' /workspace/train.log 2>/dev/null | tail -1 || echo '?/?'", check=False)
            loss  = ssh(ssh_info, "grep -oP 'loss=\\K[0-9.]+' /workspace/train.log 2>/dev/null | tail -1 || echo '?'", check=False)
            lines = ssh(ssh_info, "wc -l < /workspace/train.log 2>/dev/null || echo 0", check=False)
            print(f"  [{time.strftime('%H:%M:%S')}] Epoch {epoch}  loss={loss}  log={lines}L", end="\r")

            # Check for crash
            err = ssh(ssh_info, "tail -3 /workspace/train.log 2>/dev/null | grep -i 'error\\|traceback' || echo ''", check=False)
            if err:
                print(f"\n  ⚠️  {err}")
        except KeyboardInterrupt:
            print(f"\n  Detached. Resume with: python3 scripts/runpod_train.py --monitor {pod_id}")
            return False
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", metavar="POD_ID", help="Monitor existing pod")
    parser.add_argument("--download", metavar="POD_ID", help="Download from finished pod and terminate")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    state_file = OUTPUT_DIR / "runpod_state.json"

    # ── Download-only mode ────────────────────────────────────────────────────
    if args.download:
        pod_id = args.download
        print(f"Downloading from pod {pod_id}...")
        ssh_info = wait_for_ssh(pod_id, timeout=60)
        _download_and_deploy(ssh_info, pod_id, state_file)
        return

    # ── Monitor-only mode ─────────────────────────────────────────────────────
    if args.monitor:
        pod_id = args.monitor
        print(f"Monitoring pod {pod_id}...")
        ssh_info = wait_for_ssh(pod_id)
        done = monitor(pod_id, ssh_info)
        if done:
            _download_and_deploy(ssh_info, pod_id, state_file)
        return

    # ── Full run ──────────────────────────────────────────────────────────────
    # Resume existing pod if state file has a pod_id
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get("pod_id"):
        pod_id = state["pod_id"]
        print(f"Resuming pod {pod_id} from state file...")
    else:
        print("=== Andrew Voice Model — RunPod Training ===")
        print(f"  Image:   {DOCKER_IMAGE}")
        print(f"  GPU:     {GPU_TYPE}")
        print(f"  Epochs:  {EPOCHS}")
        print()

        print("[1/4] Generating SSH key...")
        pub_key = ensure_ssh_key()

        print("[2/4] Creating pod...")
        pod_id = create_pod(pub_key)
        state["pod_id"] = pod_id
        state["created"] = time.time()
        state_file.write_text(json.dumps(state, indent=2))
        print(f"  Pod ID: {pod_id}")

    print("[3/4] Waiting for SSH...")
    ssh_info = wait_for_ssh(pod_id)
    print(f"  {ssh_info['ip']}:{ssh_info['publicPort']}")

    # Write and launch training script
    print("[4/4] Launching training script...")
    status = ssh(ssh_info, "cat /workspace/training_status.txt 2>/dev/null || echo NONE", check=False)
    if status == "DONE":
        print("  Already complete — downloading...")
    else:
        # Write script to pod
        escaped = TRAIN_SCRIPT.replace("'", "'\\''")
        ssh(ssh_info, f"cat > /workspace/train.sh << 'EOF'\n{TRAIN_SCRIPT}\nEOF\nchmod +x /workspace/train.sh")
        ssh(ssh_info, "nohup bash /workspace/train.sh > /workspace/train.log 2>&1 &", check=False)
        print("  Script launched. Monitoring...")
        print()
        done = monitor(pod_id, ssh_info)
        if not done:
            return

    _download_and_deploy(ssh_info, pod_id, state_file)


def _download_and_deploy(ssh_info, pod_id, state_file):
    print("\nDownloading model...")
    scp_get(ssh_info, "/workspace/output/andrew-medium.onnx", OUTPUT_DIR / "andrew-medium.onnx")
    scp_get(ssh_info, "/workspace/output/andrew-medium.onnx.json", OUTPUT_DIR / "andrew-medium.onnx.json")

    size = (OUTPUT_DIR / "andrew-medium.onnx").stat().st_size / 1024 / 1024
    print(f"  ✅ andrew-medium.onnx ({size:.1f} MB)")
    print(f"  ✅ andrew-medium.onnx.json")

    print("Terminating pod...")
    terminate_pod(pod_id)
    state_file.unlink(missing_ok=True)
    print("  Pod terminated.")

    voices = Path("/opt/voice-assistant/voices")
    if voices.exists():
        import shutil
        shutil.copy(OUTPUT_DIR / "andrew-medium.onnx", voices / "andrew-medium.onnx")
        shutil.copy(OUTPUT_DIR / "andrew-medium.onnx.json", voices / "andrew-medium.onnx.json")
        print(f"  ✅ Deployed to {voices}")
        print()
        print("  To activate, update voice_assistant.py:")
        print("    model_path = '/opt/voice-assistant/voices/andrew-medium.onnx'")


if __name__ == "__main__":
    main()
