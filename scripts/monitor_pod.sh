#!/usr/bin/env bash
# monitor_pod.sh — check training, download if done, terminate pod
# Usage: ./monitor_pod.sh
#        ./monitor_pod.sh <pod_id> <ssh_host> [ssh_port]
# Exit 0 = done+downloaded, Exit 1 = still training, Exit 2 = error

set -euo pipefail

SSH_KEY="/home/pi/.ssh/runpod_piper"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STATE_FILE="$PROJECT_DIR/output/runpod_state.json"

if [ -f "$STATE_FILE" ]; then
  POD_ID=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d['pod_id'])")
  SSH_HOST=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d['ip'])")
  SSH_PORT=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d['port'])")
elif [ $# -ge 2 ]; then
  POD_ID="$1"
  SSH_HOST="$2"
  SSH_PORT="${3:-22022}"
elif [ $# -eq 1 ]; then
  echo "ERROR: pod_id provided but ssh_host is required without a state file."
  echo "Usage: $0 <pod_id> <ssh_host> [ssh_port]"
  echo "   or: ensure output/runpod_state.json exists with pod_id, ip, port fields"
  exit 2
else
  echo "ERROR: No pod config found."
  echo "Usage: $0 <pod_id> <ssh_host> [ssh_port]"
  echo "   or: ensure output/runpod_state.json exists with pod_id, ip, port fields"
  exit 2
fi
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(cat ~/.runpod_api_key 2>/dev/null)}"
OUTPUT_DIR="/home/pi/Documents/voice-model/output"
VOICES_DIR="/opt/voice-assistant/voices"

ssh_cmd() {
  ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" "$@"
}

echo "=== Piper Training Monitor ==="
echo "Pod: $POD_ID | $(date)"

# Check SSH connectivity
if ! ssh_cmd "echo ssh_ok" 2>/dev/null | grep -q ssh_ok; then
  echo "ERROR: Cannot SSH into pod"
  exit 2
fi

STATUS=$(ssh_cmd "cat /workspace/training_status.txt 2>/dev/null || echo RUNNING")
LATEST_CKPT=$(ssh_cmd "ls /workspace/training/lightning_logs/version_0/checkpoints/ 2>/dev/null | tail -1 || echo none")

echo "Status: $STATUS"
echo "Latest checkpoint: $LATEST_CKPT"

if [ "$STATUS" != "DONE" ]; then
  echo "Still training — nothing to do."
  exit 1
fi

echo "=== TRAINING COMPLETE — Downloading model ==="

# Check output files exist on pod
ssh_cmd "ls -lh /workspace/output/"

mkdir -p "$OUTPUT_DIR"
scp $SSH_OPTS -P "$SSH_PORT" "root@$SSH_HOST:/workspace/output/andrew-medium.onnx" "$OUTPUT_DIR/"
scp $SSH_OPTS -P "$SSH_PORT" "root@$SSH_HOST:/workspace/output/andrew-medium.onnx.json" "$OUTPUT_DIR/"

echo "Downloaded:"
ls -lh "$OUTPUT_DIR/andrew-medium.onnx" "$OUTPUT_DIR/andrew-medium.onnx.json"

# Deploy to voices dir
sudo cp "$OUTPUT_DIR/andrew-medium.onnx" "$VOICES_DIR/"
sudo cp "$OUTPUT_DIR/andrew-medium.onnx.json" "$VOICES_DIR/"
echo "Deployed to $VOICES_DIR"

# Terminate pod
echo "Terminating pod..."
curl -s -X POST "https://api.runpod.io/graphql" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d "{\"query\":\"mutation { podTerminate(input: { podId: \\\"$POD_ID\\\" }) }\"}" \
  | grep -o '"podTerminate":[^}]*'

echo "=== DONE === Model ready at $VOICES_DIR/andrew-medium.onnx"
exit 0
