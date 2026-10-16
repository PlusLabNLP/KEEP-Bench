#!/bin/bash
# Serve one of the 11 open-weight compressors with vLLM for run_compression.py.
#
#   bash src/compression/serve_vllm.sh <alias> [extra vllm arguments...]
#   bash src/compression/serve_vllm.sh qwen3-8b
#   bash src/compression/serve_vllm.sh qwen3-32b --tensor-parallel-size 2
#
# The alias, Hugging Face repository and served model name all come from MODELS in
# run_compression.py, so the client and the server always agree on the model id.
#
# Two flags are fixed here because the reported numbers depend on them:
#   --served-model-name      must match the id run_compression.py sends.
#   --enable-prefix-caching  the FLOPs estimates in src/computation/ charge the ICL prefix
#                            once per setting, which assumes cross-request prefix caching.
#
# Everything else depends on the hardware and is passed through, e.g.
#   --tensor-parallel-size    Qwen3-32B and Gemma-4-31B do not fit on a single 48 GB GPU.
#   --gpu-memory-utilization
#   --max-model-len           must be at least 23521 (longest prompt 22497 + 1024 output tokens).
#   --port                    run_compression.py expects http://localhost:8000/v1 unless
#                             --base-url is given.

set -euo pipefail

ALIAS="${1:?usage: bash src/compression/serve_vllm.sh <alias> [extra vllm arguments...]}"
shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r HUB_ID SERVED < <(python3 - "$HERE/run_compression.py" "$ALIAS" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('run_compression', sys.argv[1])
rc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rc
spec.loader.exec_module(rc)
model = rc.MODELS.get(sys.argv[2])
if model is None:
    sys.exit('unknown alias %r; available: %s' % (sys.argv[2], ', '.join(sorted(rc.MODELS))))
print(model.hub_id, model.served_model)
PY
)

echo "serving $HUB_ID as $SERVED"
exec vllm serve "$HUB_ID" \
  --served-model-name "$SERVED" \
  --enable-prefix-caching \
  "$@"
