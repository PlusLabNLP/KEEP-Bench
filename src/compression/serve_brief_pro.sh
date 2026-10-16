#!/bin/bash
# Serve BRIEF-Pro with vLLM for run_compression_brief.py.
#
#   bash src/compression/serve_brief_pro.sh [extra vllm arguments...]
#   bash src/compression/serve_brief_pro.sh --port 8001 --gpu-memory-utilization 0.5
#
# The checkpoint is stored in float32 and is served in bfloat16 (about 6.4 GB instead of
# 13 GB). BRIEF-Pro is built on Llama-3.2-3B. A 32k context covers the longest prompt in the
# benchmark (22497 tokens) plus the 1024-token output limit.

set -euo pipefail

exec vllm serve uclanlp/brief-pro \
  --served-model-name brief-pro \
  --dtype bfloat16 \
  --max-model-len 32768 \
  "$@"
