#!/usr/bin/env zsh
# Install pinned inference code and ready-to-run weights locally.
set -euo pipefail

tool_root=${0:A:h:h}
kimodo_commit=58e781898b3d7e328a676a75d3e338c45dce3ad9
sse2neon_tag=v1.9.1
sse2neon_sha256=78632498a57bf7e080e84cb8d74c47ce93b204cece77eab79de2908eb8bc92ed
weights_repo=Javascript/Llama-3-Kimodo-Turbo
weights_revision=bcdf39bf070269e6dbab045334c93cb47a73f033

cd "$tool_root"
./scripts/build-autorig.zsh
mkdir -p .deps out/weights
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
uv pip install --python .venv/bin/python ./.deps/kimodo 'torch==2.14.1' 'mlx==0.32.3' safetensors tokenizers 'huggingface_hub[hf_xet]==1.33.0'
uv pip install --python .venv/bin/python --no-deps -e .

export HF_HOME="$tool_root/out/hf"
.venv/bin/hf download "$weights_repo" --revision "$weights_revision" --local-dir out/weights \
  --include 'Kimodo-SOMA-RP-v1/*' --include 'llm2vec-llama3-8b/*' \
  --include 'licenses/*' --include LICENSE --include NOTICE.txt
# The installed upstream package includes its runtime code and compiled extension.
rm -rf "$tool_root/.deps"
echo "Ready. Run .venv/bin/kimodo-turbo ui or .venv/bin/kimodo-turbo gen --help"
