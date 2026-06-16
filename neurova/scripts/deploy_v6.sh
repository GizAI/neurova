#!/usr/bin/env bash
set -euo pipefail
# Neurova V6 — Deploy & Run on ml-dmc8
# Usage: bash neurova/scripts/deploy_v6.sh

HOST="ml-dmc8"
NEUROVA_DIR="/home/user/workspace/neurova"
CONDA_ENV="neurova_vsa"
SCRIPT="neurova/v6.py"

echo "=== Neurova V6 Deploy ==="

# 1. Sync code
echo "[1/4] Syncing code..."
rsync -avz --no-owner --no-group \
  "$NEUROVA_DIR/neurova/" \
  "$HOST:$NEUROVA_DIR/neurova/" 2>/dev/null || true

# Also scp as fallback
scp -r "$NEUROVA_DIR/neurova" "$HOST:$NEUROVA_DIR/" 2>/dev/null || {
    echo "[!] SCP failed, trying alternative..."
    rsync -avz "$NEUROVA_DIR/$SCRIPT" "$HOST:$NEUROVA_DIR/$SCRIPT"
}

# 2. Check env
echo "[2/4] Checking conda env..."
ssh "$HOST" "conda env list | grep $CONDA_ENV || echo 'NEED_CREATE'" 2>/dev/null || echo "ssh failed"

# 3. Install deps if needed
echo "[3/4] Installing deps..."
ssh "$HOST" "source ~/miniconda3/etc/profile.d/conda.sh && \
             (conda env list | grep $CONDA_ENV || conda create -y -n $CONDA_ENV python=3.10) && \
             conda activate $CONDA_ENV && \
             pip install -q torch transformers bitsandbytes usearch numpy sentencepiece 2>&1 | tail -3" 2>/dev/null || echo "[!] Install skipped"

# 4. Run
echo "[4/4] Starting V6..."
echo ""
echo "  To connect: ssh $HOST"
echo "  cd $NEUROVA_DIR && conda activate $CONDA_ENV && python3 $SCRIPT"
echo "  Or with KV injection disabled: V6_KV_INJECT=0 python3 $SCRIPT"
echo ""
ssh -t "$HOST" "cd $NEUROVA_DIR && source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && python3 $SCRIPT" 2>/dev/null || {
    echo ""
    echo "[!] Direct SSH failed. Run manually:"
    echo "  ssh $HOST"
    echo "  cd $NEUROVA_DIR"
    echo "  conda activate $CONDA_ENV"
    echo "  python3 $SCRIPT"
}
