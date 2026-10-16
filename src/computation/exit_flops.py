#!/usr/bin/env python3
r"""Estimate EXIT (Gemma-2B-it + rank-64 LoRA) major-matmul FLOPs.

EXIT scores one classifier prompt per candidate sentence. Two execution shapes
are supported; their costs differ by more than 3x.

Per-sentence prompts (`prompt_tokens`)
--------------------------------------
The official path: every candidate is a complete, independent prompt. For
prompt j with effective processed length p_j and vocabulary-head positions z_j:

    F_j = (3,963,617,280 + 156,893,184 u) p_j
          + 147,456 A(p_j)
          + 1,048,576,000 z_j,

where u=1 for an unmerged LoRA adapter and 0 after merging. A(p)=p(p+1)/2
under the efficient-causal convention. The official Transformers path reads
`outputs.logits[:, -1, :]` after materializing full-sequence logits, so
`logit_mode` defaults to "all"; use "last" only for a backend that projects
the last position alone.

Prefix-reusing groups (`groups`)
--------------------------------
`src/compression/run_compression_exit.py` splits each prompt at `Sentence:\n`,
forwards the shared head once per group, and runs the per-sentence suffixes on
that KV cache. One group is one document for `--context-scope document`, or one
whole row for `--context-scope full`. For a group with prefix length q and
suffix lengths t_1..t_n:

    prefix   F = L q + 147,456 A(q) + H z_prefix
    batches  F = sum over batches of
                 L s w + 147,456 s w (q + w) + H z_batch

where s is the batch row count and w its padded width. Three details follow
what the runner executes:

* The suffix block is *rectangular*: w queries attend over q + w keys, and
  right padding does not remove any of that work from the kernel. There is no
  causal triangle to halve, unlike the per-sentence path above.
* Batches are built from suffixes sorted by length, then filled greedily under
  both `batch_size` and `max_batch_tokens` (the latter caps w * s). This is
  `padding_mode="batch-max"`. `padding_mode="none"` drops padding entirely and
  gives the batch-independent lower bound: L t_j + 147,456 (t_j q + A(t_j)).
* `logit_mode="all"` charges q for the prefix forward and s*w per batch,
  because the runner passes no `logits_to_keep` and then indexes
  `out.logits[arange(size), last]`, which only exists if every position was
  projected. With a 256k vocabulary this is about 20% of the total, so "last"
  (1 per group, s per batch) is a lower bound.

Nothing amortizes across requests here: every group pays its own prefill, so
`llm_flops.py --prefix-tokens` -- which assumes one fixed prefix shared by a
whole batch -- does not apply.

Paper runs
----------
The runner scores with the checkpoint merged by
`run_compression_exit.py --stage merge`, so pass `--merged-lora` /
`"lora_unmerged": false`. The runner's defaults are `batch_size=64` and
`max_batch_tokens=8192`.

Sources:
https://github.com/ThisIsHwang/EXIT
https://huggingface.co/doubleyyh/exit-gemma-2b/blob/main/adapter_config.json
"""

from __future__ import annotations

import argparse
from typing import Any

from flops_cli import (
    add_batch_arguments,
    causal_pairs,
    fail,
    print_json,
    require_int,
    result_record,
    run_batch,
)


METHOD = "EXIT"
BASE_LINEAR_PER_TOKEN = 3_963_617_280
LORA_LINEAR_PER_TOKEN = 156_893_184
ATTENTION_PER_PAIR = 147_456
LM_HEAD_PER_POSITION = 1_048_576_000
# src/compression/run_compression_exit.py defaults.
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_BATCH_TOKENS = 8192
GROUP_PADDING_MODES = ("batch-max", "none")


def _int_list(value: Any, name: str, *, minimum: int) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list of integers")
    return [require_int(item, f"{name}[{i}]", minimum=minimum) for i, item in enumerate(value)]


def _effective_lengths(
    prompt_tokens: list[int],
    *,
    batch_size: int | None,
    explicit_padded_tokens: list[int] | None,
) -> list[int]:
    if explicit_padded_tokens is not None:
        if len(explicit_padded_tokens) != len(prompt_tokens):
            raise ValueError("padded_prompt_tokens must match prompt_tokens in length")
        for i, (raw, padded) in enumerate(zip(prompt_tokens, explicit_padded_tokens)):
            if padded < raw:
                raise ValueError(
                    f"padded_prompt_tokens[{i}]={padded} is shorter than prompt_tokens[{i}]={raw}"
                )
        return explicit_padded_tokens

    if batch_size is None:
        return prompt_tokens

    effective: list[int] = []
    for start in range(0, len(prompt_tokens), batch_size):
        group = prompt_tokens[start : start + batch_size]
        effective.extend([max(group)] * len(group))
    return effective


def suffix_batches(
    suffix_tokens: list[int], *, batch_size: int, max_batch_tokens: int
) -> list[list[int]]:
    """Reproduce score_grouped(): sort by suffix length, then fill greedily.

    Returns a list of batches, each a list of suffix lengths. The two caps are
    the row count and the padded token count w * s.
    """
    order = sorted(range(len(suffix_tokens)), key=lambda i: suffix_tokens[i])
    batches: list[list[int]] = []
    batch: list[int] = []
    longest = 0
    for i in order:
        length = suffix_tokens[i]
        nxt = max(longest, length)
        if batch and (
            len(batch) + 1 > batch_size or nxt * (len(batch) + 1) > max_batch_tokens
        ):
            batches.append(batch)
            batch, longest = [length], length
        else:
            batch.append(length)
            longest = nxt
    if batch:
        batches.append(batch)
    return batches


def _grouped_calculate(row: dict[str, Any], *, linear_per_token: int) -> dict[str, Any]:
    groups = row["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a non-empty JSON list")

    padding_mode = row.get("padding_mode", "batch-max")
    if padding_mode not in GROUP_PADDING_MODES:
        raise ValueError(f"padding_mode must be one of {GROUP_PADDING_MODES}")
    batch_size = require_int(
        row.get("batch_size", DEFAULT_BATCH_SIZE), "batch_size", minimum=1
    )
    max_batch_tokens = require_int(
        row.get("max_batch_tokens", DEFAULT_MAX_BATCH_TOKENS),
        "max_batch_tokens",
        minimum=1,
    )
    logit_mode = row.get("logit_mode", "all")
    if logit_mode not in {"all", "last"}:
        raise ValueError("logit_mode must be 'all' or 'last'")

    prefix_linear = prefix_attention = prefix_lm = 0
    suffix_linear = suffix_attention = suffix_lm = 0
    num_sentences = num_batches = 0
    real_suffix_tokens = padded_suffix_tokens = 0

    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{index}] must be a JSON object")
        prefix_tokens = require_int(
            group["prefix_tokens"], f"groups[{index}].prefix_tokens", minimum=1
        )
        suffix_tokens = _int_list(
            group["suffix_tokens"], f"groups[{index}].suffix_tokens", minimum=1
        )
        if not suffix_tokens:
            raise ValueError(f"groups[{index}].suffix_tokens must be non-empty")
        if max_batch_tokens < max(suffix_tokens):
            raise ValueError(
                f"max_batch_tokens ({max_batch_tokens}) is below the longest suffix "
                f"in groups[{index}] ({max(suffix_tokens)}); no batch could be formed"
            )

        num_sentences += len(suffix_tokens)
        real_suffix_tokens += sum(suffix_tokens)

        # The prefix is forwarded once, as a single causal sequence.
        prefix_linear += linear_per_token * prefix_tokens
        prefix_attention += ATTENTION_PER_PAIR * causal_pairs(prefix_tokens)
        prefix_lm += LM_HEAD_PER_POSITION * (prefix_tokens if logit_mode == "all" else 1)

        if padding_mode == "none":
            num_batches += len(suffix_tokens)
            padded_suffix_tokens += sum(suffix_tokens)
            for length in suffix_tokens:
                suffix_linear += linear_per_token * length
                suffix_attention += ATTENTION_PER_PAIR * (
                    length * prefix_tokens + causal_pairs(length)
                )
                suffix_lm += LM_HEAD_PER_POSITION * (length if logit_mode == "all" else 1)
            continue

        for batch in suffix_batches(
            suffix_tokens, batch_size=batch_size, max_batch_tokens=max_batch_tokens
        ):
            rows_in_batch = len(batch)
            width = max(batch)
            positions = rows_in_batch * width
            num_batches += 1
            padded_suffix_tokens += positions
            suffix_linear += linear_per_token * positions
            # Rectangular block: width queries over prefix_tokens + width keys.
            suffix_attention += ATTENTION_PER_PAIR * positions * (prefix_tokens + width)
            suffix_lm += LM_HEAD_PER_POSITION * (
                positions if logit_mode == "all" else rows_in_batch
            )

    linear_flops = prefix_linear + suffix_linear
    attention_flops = prefix_attention + suffix_attention
    lm_head_flops = prefix_lm + suffix_lm
    return {
        "flops": linear_flops + attention_flops + lm_head_flops,
        "input_shape": "groups",
        "num_groups": len(groups),
        "num_candidates": num_sentences,
        "num_suffix_batches": num_batches,
        "padding_mode": padding_mode,
        "batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
        "logit_mode": logit_mode,
        "real_suffix_tokens": real_suffix_tokens,
        "padded_suffix_tokens": padded_suffix_tokens,
        "suffix_padding_overhead_tokens": padded_suffix_tokens - real_suffix_tokens,
        "prefix_flops": prefix_linear + prefix_attention + prefix_lm,
        "suffix_flops": suffix_linear + suffix_attention + suffix_lm,
        "linear_flops": linear_flops,
        "attention_flops": attention_flops,
        "lm_head_flops": lm_head_flops,
    }


def _per_sentence_calculate(
    row: dict[str, Any], *, linear_per_token: int
) -> dict[str, Any]:
    prompt_tokens = _int_list(row["prompt_tokens"], "prompt_tokens", minimum=1)
    if not prompt_tokens:
        raise ValueError("prompt_tokens must contain at least one EXIT prompt")

    batch_size_value = row.get("batch_size")
    batch_size = (
        None
        if batch_size_value is None
        else require_int(batch_size_value, "batch_size", minimum=1)
    )
    explicit_padded = row.get("padded_prompt_tokens")
    padded_prompt_tokens = (
        None
        if explicit_padded is None
        else _int_list(explicit_padded, "padded_prompt_tokens", minimum=1)
    )
    effective_tokens = _effective_lengths(
        prompt_tokens,
        batch_size=batch_size,
        explicit_padded_tokens=padded_prompt_tokens,
    )

    attention_mode = row.get("attention_mode", "efficient-causal")
    if attention_mode not in {"efficient-causal", "dense-masked"}:
        raise ValueError(
            "attention_mode must be 'efficient-causal' or 'dense-masked'"
        )
    if attention_mode == "efficient-causal":
        attention_pairs = [causal_pairs(length) for length in effective_tokens]
    else:
        attention_pairs = [length * length for length in effective_tokens]

    explicit_logits = row.get("logit_positions")
    if explicit_logits is not None:
        logit_positions = _int_list(
            explicit_logits, "logit_positions", minimum=0
        )
        if len(logit_positions) != len(prompt_tokens):
            raise ValueError("logit_positions must match prompt_tokens in length")
        logit_mode = "explicit"
    else:
        logit_mode = row.get("logit_mode", "all")
        if logit_mode == "all":
            logit_positions = effective_tokens
        elif logit_mode == "last":
            logit_positions = [1] * len(prompt_tokens)
        else:
            raise ValueError("logit_mode must be 'all' or 'last'")

    linear_flops = linear_per_token * sum(effective_tokens)
    attention_flops = ATTENTION_PER_PAIR * sum(attention_pairs)
    lm_head_flops = LM_HEAD_PER_POSITION * sum(logit_positions)
    return {
        "flops": linear_flops + attention_flops + lm_head_flops,
        "input_shape": "prompt_tokens",
        "prompt_tokens": prompt_tokens,
        "effective_prompt_tokens": effective_tokens,
        "num_candidates": len(prompt_tokens),
        "batch_size": batch_size,
        "attention_mode": attention_mode,
        "attention_pairs": attention_pairs,
        "logit_mode": logit_mode,
        "logit_positions": logit_positions,
        "linear_flops": linear_flops,
        "attention_flops": attention_flops,
        "lm_head_flops": lm_head_flops,
    }


def calculate(row: dict[str, Any]) -> dict[str, Any]:
    has_groups = "groups" in row
    has_prompts = "prompt_tokens" in row
    if has_groups == has_prompts:
        raise ValueError("provide exactly one of 'groups' or 'prompt_tokens'")

    lora_unmerged = row.get("lora_unmerged", True)
    if not isinstance(lora_unmerged, bool):
        raise ValueError("lora_unmerged must be true or false")
    linear_per_token = BASE_LINEAR_PER_TOKEN + (
        LORA_LINEAR_PER_TOKEN if lora_unmerged else 0
    )

    details = (
        _grouped_calculate(row, linear_per_token=linear_per_token)
        if has_groups
        else _per_sentence_calculate(row, linear_per_token=linear_per_token)
    )
    details["lora_unmerged"] = lora_unmerged
    return details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate EXIT major-matmul inference FLOPs."
    )
    add_batch_arguments(parser)
    parser.add_argument("--prompt-tokens", type=int, nargs="+")
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        help="Shared head length of one prefix-reusing group; needs --suffix-tokens.",
    )
    parser.add_argument(
        "--suffix-tokens",
        type=int,
        nargs="+",
        help="Per-sentence suffix lengths of one prefix-reusing group.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "With --prompt-tokens, model sequential batches padded to each batch "
            f"maximum. With --prefix-tokens, the runner's row cap (default "
            f"{DEFAULT_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        help=f"Padded-token cap w*s for a group (default {DEFAULT_MAX_BATCH_TOKENS}).",
    )
    parser.add_argument(
        "--padding-mode",
        choices=GROUP_PADDING_MODES,
        help="Group path only: 'batch-max' as executed, 'none' for the unpadded bound.",
    )
    parser.add_argument(
        "--padded-prompt-tokens",
        type=int,
        nargs="+",
        help="Optional exact processed length for every prompt; overrides batch-size padding.",
    )
    parser.add_argument(
        "--merged-lora",
        action="store_true",
        help="Exclude LoRA A/B matmuls because the adapter was merged into base weights.",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("efficient-causal", "dense-masked"),
        default="efficient-causal",
        help="Per-sentence path only; the group path is always a rectangular block.",
    )
    parser.add_argument(
        "--logit-mode",
        choices=("all", "last"),
        default="all",
        help="'all' matches the official Transformers path; 'last' models an optimized backend.",
    )
    parser.add_argument(
        "--logit-positions",
        type=int,
        nargs="+",
        help="Per-sentence path only: exact per-prompt vocabulary-head position counts.",
    )
    args = parser.parse_args()

    try:
        if args.input_jsonl:
            if args.prompt_tokens is not None or args.prefix_tokens is not None:
                raise ValueError("use either --input-jsonl or single-case arguments")
            print_json(
                run_batch(
                    method=METHOD,
                    input_path=args.input_jsonl,
                    output_path=args.output_jsonl,
                    calculator=calculate,
                )
            )
            return

        if args.output_jsonl:
            raise ValueError("--output-jsonl requires --input-jsonl")

        row: dict[str, Any] = {
            "lora_unmerged": not args.merged_lora,
            "logit_mode": args.logit_mode,
        }
        if args.prefix_tokens is not None:
            if args.prompt_tokens is not None:
                raise ValueError(
                    "use either --prompt-tokens or --prefix-tokens/--suffix-tokens"
                )
            if args.suffix_tokens is None:
                raise ValueError("--prefix-tokens requires --suffix-tokens")
            row["groups"] = [
                {
                    "prefix_tokens": args.prefix_tokens,
                    "suffix_tokens": args.suffix_tokens,
                }
            ]
            if args.padding_mode is not None:
                row["padding_mode"] = args.padding_mode
            if args.batch_size is not None:
                row["batch_size"] = args.batch_size
            if args.max_batch_tokens is not None:
                row["max_batch_tokens"] = args.max_batch_tokens
        else:
            if args.suffix_tokens is not None:
                raise ValueError("--suffix-tokens requires --prefix-tokens")
            if args.prompt_tokens is None:
                raise ValueError(
                    "single-case mode requires --prompt-tokens, or "
                    "--prefix-tokens with --suffix-tokens"
                )
            row["prompt_tokens"] = args.prompt_tokens
            row["attention_mode"] = args.attention_mode
            if args.batch_size is not None:
                row["batch_size"] = args.batch_size
            if args.padded_prompt_tokens is not None:
                row["padded_prompt_tokens"] = args.padded_prompt_tokens
            if args.logit_positions is not None:
                row["logit_positions"] = args.logit_positions
        print_json(result_record(METHOD, row, calculate(row)))
    except (KeyError, TypeError, ValueError) as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()
