#!/usr/bin/env bash
set -e

# 환경 변수 및 경로
CONDA_ENV="neurova_vsa"
MODELS_DIR="$HOME/tabby-models/models"
TABBY_DIR="$HOME/tabby-models/tabbyAPI"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

echo "=== 4. 모델 Config 및 Chat Template 패치 ==="
SCRIPT_DIR="$SCRIPT_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

model_dir = Path.home() / "tabby-models/models/Qwen3.6-27B-exl3-3.08bpw"
script_dir = Path(os.environ["SCRIPT_DIR"])
local_template_path = script_dir / "templates/qwen3_coder_neurova_chat_template.jinja"

config_path = model_dir / "config.json"
cfg = json.loads(config_path.read_text())
cfg["architectures"] = ["Qwen3_5ForCausalLM"]
for key, value in cfg.get("text_config", {}).items():
    cfg[key] = value
config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=4) + "\n")

local_template = local_template_path.read_text() if local_template_path.exists() else None
old = "{%- if enable_thinking is defined and enable_thinking is false %}"
new = "{%- if enable_thinking is not defined or enable_thinking is false %}"
guard = "Final answer must be in Korean unless the user asks otherwise. Do not include Chinese text in final answer. Never end with a promise of future action; if work remains, call tools in the same reply, otherwise provide completed results."
tools_marker = """    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\\n\\n' + content }}
        {%- endif %}
    {%- endif %}
"""
tools_replacement = """    {%- set neurova_lang_guard = 'Final answer must be in Korean unless the user asks otherwise. Do not include Chinese text in final answer. Never end with a promise of future action; if work remains, call tools in the same reply, otherwise provide completed results.' %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\\n\\n' + neurova_lang_guard + '\\n' + content }}
        {%- else %}
            {{- '\\n\\n' + neurova_lang_guard }}
        {%- endif %}
    {%- else %}
        {{- '\\n\\n' + neurova_lang_guard }}
    {%- endif %}
"""
plain_marker = """    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}
    {%- endif %}
"""
plain_replacement = """    {%- set neurova_lang_guard = 'Final answer must be in Korean unless the user asks otherwise. Do not include Chinese text in final answer. Never end with a promise of future action; if work remains, call tools in the same reply, otherwise provide completed results.' %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {{- '<|im_start|>system\\n' + neurova_lang_guard + '\\n' + content + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\n' + neurova_lang_guard + '<|im_end|>\\n' }}
    {%- endif %}
"""
for name in ("chat_template.jinja", "tokenizer_config.json"):
    path = model_dir / name
    if not path.exists():
        continue
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if local_template is not None:
            data["chat_template"] = local_template
        else:
            template = data.get("chat_template", "")
            if old in template:
                data["chat_template"] = template.replace(old, new, 1)
            if guard not in data.get("chat_template", ""):
                data["chat_template"] = data["chat_template"].replace(tools_marker, tools_replacement, 1).replace(plain_marker, plain_replacement, 1)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    else:
        if local_template is not None:
            template = local_template
        else:
            template = path.read_text()
            if old in template:
                template = template.replace(old, new, 1)
            if guard not in template:
                template = template.replace(tools_marker, tools_replacement, 1).replace(plain_marker, plain_replacement, 1)
        path.write_text(template)
PY

echo "=== 5. TabbyAPI 소스 설치 및 설정 ==="
cd $HOME/tabby-models
if [ ! -d "tabbyAPI" ]; then
    git clone https://github.com/theroyallab/tabbyAPI.git
fi
cd tabbyAPI
pip install -r requirements.txt || true
pip install -e .

echo "=== 6. ExLlamaV3 byte-fallback streaming decode 패치 ==="
python3 - <<'PY'
from pathlib import Path

path = Path("backends/exllamav3/model.py")
text = path.read_text()
if "response_token_ids = []" not in text:
    text = text.replace(
        '        generated_tokens = 0\n        full_response = ""\n        metrics_result = {}\n',
        '        generated_tokens = 0\n        full_response = ""\n        response_token_ids = []\n        metrics_result = {}\n',
        1,
    )
    old = '''                chunk = unwrap(result.get("text"), "")
                if chunk:
                    chunk_tokens = result.get("token_ids", self.tokenizer.encode(chunk))
                    full_response += chunk

                    # Extract token IDs as a plain list for downstream consumers
                    if isinstance(chunk_tokens, torch.Tensor):
                        token_id_list = chunk_tokens.flatten().tolist()
                        generated_tokens += chunk_tokens.size(dim=0)
                    elif isinstance(chunk_tokens, tuple):
                        first = chunk_tokens[0]
                        if isinstance(first, torch.Tensor):
                            token_id_list = first.flatten().tolist()
                        else:
                            token_id_list = list(first)
                        generated_tokens += len(token_id_list)
                    else:
                        token_id_list = list(chunk_tokens)
                        generated_tokens += len(token_id_list)
'''
    new = '''                raw_chunk = unwrap(result.get("text"), "")
                chunk_tokens = result.get("token_ids")

                # ExLlamaV3 may stream byte-fallback tokens whose individual text
                # fragments are invalid UTF-8 and decode as U+FFFD. Decode from the
                # accumulated token IDs instead of trusting result["text"].
                if chunk_tokens is not None:
                    if isinstance(chunk_tokens, torch.Tensor):
                        token_id_list = chunk_tokens.flatten().tolist()
                    elif isinstance(chunk_tokens, tuple):
                        first = chunk_tokens[0]
                        if isinstance(first, torch.Tensor):
                            token_id_list = first.flatten().tolist()
                        else:
                            token_id_list = list(first)
                    else:
                        token_id_list = list(chunk_tokens)

                    generated_tokens += len(token_id_list)
                    response_token_ids.extend(token_id_list)
                    decoded_response = self.tokenizer.decode(torch.tensor([response_token_ids]))[0]
                    stable_response = decoded_response.split("\\ufffd", 1)[0]
                    chunk = stable_response[len(full_response) :]
                else:
                    chunk = raw_chunk
                    token_id_list = self.tokenizer.encode(chunk) if chunk else []
                    generated_tokens += len(token_id_list)

                if chunk:
                    full_response += chunk
'''
    if old not in text:
        raise SystemExit("expected ExLlamaV3 generate block not found")
    text = text.replace(old, new, 1)
    path.write_text(text)
PY

cat > config.yml << "CONFIG_EOF"
model:
  model_dir: /home/user/tabby-models/models
  model_name: Qwen3.6-27B-exl3-3.08bpw
  max_seq_len: 100352
  cache_size: 100352
  cache_mode: "3,2"
  chunk_size: 256
  output_chunking: true
  reasoning: true
  reasoning_start_token: "<think>"
  reasoning_end_token: "</think>"
  tool_format: qwen3_coder
  max_batch_size: 2

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
