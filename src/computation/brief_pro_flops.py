#!/usr/bin/env python3
"""Estimate BRIEF-Pro major-matmul inference FLOPs for one or many cases.

The released BRIEF-Pro checkpoint uses a Llama-3.2-3B-Instruct backbone.

Formula (one multiply-add = 2 FLOPs, KV cache, efficient causal attention):
    T = P + G - 1
    A(T) = T(T + 1) / 2
    F = 5,637,144,576 T + 344,064 A(T) + 788,004,864 Z

P is the complete BRIEF-Pro-tokenized prompt length, G is the actual generated
token-ID count, and Z is the number of vocabulary-head positions (normally G).
Norms, RoPE, activations, softmax, sampling, and communication are excluded.

Architecture source:
https://huggingface.co/uclanlp/brief-pro/blob/main/config.json
"""

from __future__ import annotations

from typing import Any

from flops_cli import causal_pairs, generation_lengths, run_generation_cli


METHOD = "BRIEF-Pro"
LINEAR_PER_TOKEN = 5_637_144_576
ATTENTION_PER_PAIR = 344_064
LM_HEAD_PER_POSITION = 788_004_864


def calculate(row: dict[str, Any]) -> dict[str, Any]:
    prompt, generated, logits, processed = generation_lengths(row)
    pairs = causal_pairs(processed)
    linear_flops = LINEAR_PER_TOKEN * processed
    attention_flops = ATTENTION_PER_PAIR * pairs
    lm_head_flops = LM_HEAD_PER_POSITION * logits
    return {
        "flops": linear_flops + attention_flops + lm_head_flops,
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "processed_tokens": processed,
        "logit_positions": logits,
        "causal_attention_pairs": pairs,
        "linear_flops": linear_flops,
        "attention_flops": attention_flops,
        "lm_head_flops": lm_head_flops,
    }


if __name__ == "__main__":
    run_generation_cli(
        method=METHOD,
        description="Estimate BRIEF-Pro major-matmul inference FLOPs.",
        calculator=calculate,
    )
