# KEEP-Bench

Code for **KEEP-Bench: Evaluating Evidence Preservation in Context Compression**.

KEEP-Bench tests whether a context compressor keeps the evidence needed to answer a question. Each of its 4,102 cases is annotated with the evidence propositions a faithful compression must retain, and an LLM judge checks every proposition directly in the compressed context. The cases cover three evidence structures: parallel evidence (two valid interpretations), conditional evidence (a claim that holds only under a qualifier) and imbalanced evidence (dominant and minority answer perspectives).

- Paper: coming soon
- Dataset: [uclanlp/KEEP-Bench](https://huggingface.co/datasets/uclanlp/KEEP-Bench)
- Website: coming soon

## Repository structure

```
scripts/prepare_data.py        download the dataset and write data/
data/icl/                      in-context demonstrations used by the LLM compressors
src/compression/               compressors: open-weight LLMs (vLLM), GPT-5.4 mini, Claude Haiku 4.5,
                               BRIEF-Pro, EXIT, RECOMP; ICL x thinking ablation; position experiment
src/evaluation/                judge inputs, LLM judge, scoring
src/computation/               inference FLOPs per method
src/data/                      position-controlled variants of the two-hop parallel cases
results/flops_1x.json          FLOPs per case at the default budget, used for EPR
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm      # sentence splitting for EXIT and RECOMP (extractive)
```

EXIT, RECOMP and the LoRA merge for EXIT also need PyTorch and peft (`pip install -r requirements-gpu.txt`). Open-weight LLMs and BRIEF-Pro are served with [vLLM](https://github.com/vllm-project/vllm), which is not in the requirements files; install it in a separate environment and start the server from there. Gemma-4 needs a newer vLLM than the other models, so we used two environments:

```bash
# Qwen3, Llama-3 and BRIEF-Pro
pip install vllm==0.8.5

# Gemma-4
pip install vllm==0.29.0
```

The Gemma-4 results in the paper were produced with a vLLM development build (0.28.1rc1.dev87, commit 6d4562c59) with transformers 5.16.1 and torch 2.13.0 (CUDA 12.9); 0.29.0 is the first release after that build.

The dataset is gated on Hugging Face. Accept its terms on the dataset page, log in, and prepare the data:

```bash
hf auth login
python scripts/prepare_data.py
```

This writes one file per subset under `data/` (`data/parallel/two_hop.jsonl`, ..., `data/policy/four_perspectives.jsonl`), each with a `.paragraphs.jsonl` sibling holding the structured paragraphs.

## Compression

Every compressor reads the same inputs and writes one file per subset to `outputs/<method>/<thinking>/<budget>/<subset>.jsonl`, where each row is the input row plus `compressed_context`. Sentence budgets are `0.5x`, `1x` and `2x`; `1x` is the default budget used for the main results (K = 2h sentences for h-hop parallel and conditional cases, K = 2n + 1 for imbalanced cases with n perspectives).

Open-weight LLMs (Qwen3, Llama-3, Gemma-4) are served with vLLM and prompted with the demonstrations in `data/icl/`:

```bash
# in the vLLM environment
bash src/compression/serve_vllm.sh qwen3-8b

# in the KEEP-Bench environment, once the server is up
python src/compression/run_compression.py --model qwen3-8b                 # all subsets and budgets
python src/compression/run_compression.py --model qwen3-8b --budget 1x     # default budget only
python src/compression/run_compression.py --list-models
```

Proprietary LLMs:

```bash
export OPENAI_API_KEY=...
python src/compression/run_compression_gpt.py
export ANTHROPIC_API_KEY=...
python src/compression/run_compression_claude.py
```

Dedicated compressors:

```bash
bash src/compression/serve_brief_pro.sh               # in the vLLM environment
python src/compression/run_compression_brief.py

python src/compression/run_compression_exit.py --stage merge
python src/compression/run_compression_exit.py --stage split
python src/compression/run_compression_exit.py --stage score --context-scope document

python src/compression/run_compression_recomp_extr.py --stage split
python src/compression/run_compression_recomp_extr.py --stage score
python src/compression/run_compression_recomp_abst.py
```

RECOMP (abstractive) has no length control and writes a single operating point under the budget directory `fixed`. Each script documents its options in its module docstring and `--help`.

## Evaluation

Evaluation has three steps: expand compressed outputs into one judge request per evidence proposition, run the judge, and aggregate the judgements.

```bash
python src/evaluation/build_entailment_check_input.py
export OPENAI_API_KEY=...
python src/evaluation/run_entailment_check.py --dry-run
python src/evaluation/run_entailment_check.py
python src/evaluation/score.py --flops results/flops_1x.json
```

`score.py` reports Para-Rec, Cond-Rec and All-Rec per setting, overall All-Rec over all 4,102 cases, and EPR for default-budget runs with a FLOPs estimate. The judge is GPT-5.2 with reasoning effort set to high. It issues one request per evidence proposition: 15,953 requests for one method at one budget. Scores obtained with a different judge are not directly comparable to those in the paper.

### Evaluating a new compressor

Write one output file per subset, row-aligned with the files in `data/`, keeping every input field and adding `compressed_context`:

```python
import json
from pathlib import Path

SUBSETS = {
    'parallel/two_hop': 'parallel_setting_two_hop',
    'parallel/three_hop': 'parallel_setting_three_hop',
    'parallel/four_hop': 'parallel_setting_four_hop',
    'conditional/two_hop': 'conditional_setting_two_hop',
    'conditional/three_hop': 'conditional_setting_three_hop',
    'conditional/four_hop': 'conditional_setting_four_hop',
    'science/three_perspectives': 'science_domain_three_perspectives',
    'science/four_perspectives': 'science_domain_four_perspectives',
    'policy/four_perspectives': 'policy_domain',
}

out_dir = Path('outputs/my-method/without_thinking/1x')
out_dir.mkdir(parents=True, exist_ok=True)
for subset, stem in SUBSETS.items():
    with open(f'data/{subset}.jsonl') as src, open(out_dir / f'{stem}.jsonl', 'w') as dst:
        for line in src:
            row = json.loads(line)
            row['compressed_context'] = my_compressor(row['question'], row['context'])
            dst.write(json.dumps(row, ensure_ascii=False) + '\n')
```

Then run the three evaluation steps above, optionally restricted with `--only my-method`.

## Analyses

- Budgets: run any compressor with `--budget 0.5x`, `1x` or `2x`.
- ICL x thinking ablation (Gemma-4): `python src/compression/run_ablation.py --model gemma4-e4b`.
- Evidence position: build the variants, compress, judge and score them.

  ```bash
  python src/data/build_position_variants.py
  python src/compression/run_compression_position.py --model qwen3-8b
  python src/evaluation/build_entailment_check_input.py --stem 'position_*'
  python src/evaluation/run_entailment_check.py
  python src/evaluation/score_position.py
  ```

- FLOPs: `python src/computation/run_flops.py` estimates the average inference FLOPs per case from the actual prompts and outputs. `results/flops_1x.json` holds the values used in the paper.

## License

The code is released under the MIT License. The dataset and the demonstrations in `data/icl/` are released under CC BY-NC 4.0; see the [dataset card](https://huggingface.co/datasets/uclanlp/KEEP-Bench) for the terms of use.

## Citation

```bibtex
@article{man2026keepbench,
  title   = {{KEEP-Bench}: Evaluating Evidence Preservation in Context Compression},
  author  = {Man, Ziqing and Wu, Di and Gu, Jia-Chen and Zhang, Junyi and Chang, Kai-Wei and Peng, Nanyun},
  journal = {arXiv preprint arXiv:TBD},
  year    = {2026}
}
```
