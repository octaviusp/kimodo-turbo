#!/usr/bin/env zsh
# Reproducible, folder-local setup: pinned + patched kimodo source, Python deps in .venv, model
# weights in out/hf, and the streamed text-encoder shards in out/weights. Safe to re-run.
# The one-time `kimodo-turbo convert` loads kimodo's own bf16 encoder (~17 GB RAM, ~1 min).
set -euo pipefail

tool_root=${0:A:h:h}
kimodo_commit=58e781898b3d7e328a676a75d3e338c45dce3ad9
sse2neon_tag=v1.9.1
sse2neon_sha256=78632498a57bf7e080e84cb8d74c47ce93b204cece77eab79de2908eb8bc92ed
llama_mirror=NousResearch/Meta-Llama-3-8B-Instruct

cd "$tool_root"
mkdir -p .deps out/hf out/logs

if [[ ! -d .deps/kimodo/.git ]]; then
  git clone --quiet https://github.com/nv-tlabs/kimodo.git .deps/kimodo
fi
git -C .deps/kimodo checkout --quiet --force "$kimodo_commit"
git -C .deps/kimodo clean --quiet -fdx
for patch in "$tool_root"/patches/*.patch; do
  git -C .deps/kimodo apply "$patch"
done

header=.deps/kimodo/MotionCorrection/src/cpp/Math/sse2neon.h
curl -fsSL "https://raw.githubusercontent.com/DLTcollab/sse2neon/${sse2neon_tag}/sse2neon.h" -o "$header"
actual=$(shasum -a 256 "$header" | cut -d' ' -f1)
if [[ "$actual" != "$sse2neon_sha256" ]]; then
  echo "sse2neon.h checksum mismatch: $actual" >&2
  exit 1
fi

[[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python ./.deps/kimodo torch mlx safetensors tokenizers 'huggingface_hub[hf_xet]'
uv pip install --python .venv/bin/python --no-deps -e .

export HF_HOME="$tool_root/out/hf"
.venv/bin/hf download nvidia/Kimodo-SOMA-RP-v1
.venv/bin/hf download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
.venv/bin/hf download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
# Ungated mirror; verify_llama.py checks its weights are byte-identical to meta-llama's release.
.venv/bin/hf download "$llama_mirror" --include '*.safetensors' --include '*.json'
.venv/bin/python -I "$tool_root/scripts/verify_llama.py"

if [[ ! -f out/weights/llm2vec-llama3-8b/manifest.json ]]; then
  HF_HUB_OFFLINE=1 .venv/bin/kimodo-turbo convert
fi
