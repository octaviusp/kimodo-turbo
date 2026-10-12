"""Kimodo motion generation on MLX: streamed bf16 LLM2Vec encoder and fp32 denoiser."""

import os

# MLX 0.32 runs fp32 GEMMs as TF32 by default. Measured on the Kimodo denoiser: TF32 gives a
# 7.5e-2 max relative error against an FP64 oracle versus 1.3e-4 for full FP32 (torch FP32 CPU:
# 2.1e-4). The precision contract is FP32, so TF32 is disabled before MLX is imported.
os.environ["MLX_ENABLE_TF32"] = "0"
