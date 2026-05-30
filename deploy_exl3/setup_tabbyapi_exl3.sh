#!/usr/bin/env bash
set -e

# 환경 변수 및 경로
CONDA_ENV="neurova_vsa"
MODELS_DIR="$HOME/tabby-models/models"
TABBY_DIR="$HOME/tabby-models/tabbyAPI"

echo "=== 1. Conda 환경 활성화 ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "=== 2. ExLlamaV3 Pre-built Wheel 설치 (JIT 방지) ==="
pip uninstall -y exllamav3 pybind11 tabbyAPI 2>/dev/null || true
cd /tmp
wget -qO exllamav3.whl "https://github.com/turboderp-org/exllamav3/releases/download/v0.0.38/exllamav3-0.0.38%2Bcu128.torch2.10.0-cp310-cp310-linux_x86_64.whl"
pip install pybind11
pip install ./exllamav3.whl

echo "=== 3. 모델 가중치 다운로드 ==="
mkdir -p $MODELS_DIR
# Target 모델
if [ ! -d "$MODELS_DIR/Qwen3.6-27B-exl3-3.08bpw" ]; then
    hf download UnstableLlama/Qwen3.6-27B-exl3-3.08bpw --local-dir $MODELS_DIR/Qwen3.6-27B-exl3-3.08bpw
fi
# Draft 모델 (DFlash - 3.00bpw 브랜치 중요!)
if [ ! -d "$MODELS_DIR/Qwen3.6-27B-DFlash-exl3" ]; then
    hf download turboderp/Qwen3.6-27B-DFlash-exl3 --revision "3.00bpw" --local-dir $MODELS_DIR/Qwen3.6-27B-DFlash-exl3
fi

echo "=== 4. 모델 Config.json 패치 (호환성 수정) ==="
python3 -c "
import json, os
path = os.path.expanduser(~/tabby-models/models/Qwen3.6-27B-exl3-3.08bpw/config.json)
with open(path, r) as f: cfg = json.load(f)
cfg[architectures] = [Qwen3_5ForCausalLM]
for k, v in cfg.get(text_config, {}).items(): cfg[k] = v
with open(path, w) as f: json.dump(cfg, f, indent=4)
"

echo "=== 5. TabbyAPI 소스 설치 및 설정 ==="
cd $HOME/tabby-models
if [ ! -d "tabbyAPI" ]; then
    git clone https://github.com/theroyallab/tabbyAPI.git
fi
cd tabbyAPI
pip install -r requirements.txt || true
pip install -e .

cat > config.yml << "CONFIG_EOF"
model:
  model_dir: /home/user/tabby-models/models
  model_name: Qwen3.6-27B-exl3-3.08bpw
  max_seq_len: 147456
  cache_size: 147456
  cache_mode: "3,2"
  chunk_size: 256
  output_chunking: true
  max_batch_size: 1

draft_model:
  draft_model_dir: /home/user/tabby-models/models
  draft_model_name: Qwen3.6-27B-DFlash-exl3
  draft_cache_mode: "Q4"
  draft_num_tokens: 6

network:
  host: 0.0.0.0
  port: 5000
CONFIG_EOF

echo "=== Setup Completed ==="
