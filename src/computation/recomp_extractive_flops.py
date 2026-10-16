#!/usr/bin/env python3
"""Estimate RECOMP-extractive major-matmul inference FLOPs.

The released compressor is a 12-layer Contriever/BERT encoder. For a tokenized
sequence of length l:
    E(l) = 169,869,312 l + 36,864 l^2.

The l^2 term is deliberate: the encoder is bidirectional, so every position
attends to every position and there is no causal triangle to halve. There is no
vocabulary head.

For one query sequence l_0 and M candidate-sentence sequences l_1..l_M:
    F = sum_{j=0..M} E(l_j) + 1,536 M.

The final term counts query/sentence embedding dot products. Mean pooling and
other elementwise operations are excluded.

Sequence lengths
----------------
`max_sequence_length` (default 512, the checkpoint's `max_position_embeddings`)
clamps every sequence before padding, because the tokenizer is called with
`truncation=True` and the encoder cannot process longer inputs. Sequences are
clamped first and padded second, as in the runner.

`padding_mode` selects how padding is modeled:

    none        unpadded; the batch-independent lower bound.
    batch-max   consecutive batches of `batch_size` candidates in the given
                order, each padded to its own maximum. This is what
                `src/compression/run_compression_recomp_extr.py` does: slices
                of 32 taken in document order, `padding=True`, no sorting by
                length. The order of `candidate_tokens` changes the result, so
                pass the sentences in the order the runner saw them.
    case-max    one padded batch over the query and every candidate, which is
                the official single-batch path.

Sources:
https://huggingface.co/fangyuan/hotpotqa_extractive_compressor/blob/main/config.json
https://github.com/carriex/recomp
"""

from __future__ import annotations

import argparse
from typing import Any

from flops_cli import (
    add_batch_arguments,
    fail,
    print_json,
    require_int,
    result_record,
    run_batch,
)


METHOD = "RECOMP-extractive"
ENCODER_LINEAR_PER_TOKEN = 169_869_312
ENCODER_ATTENTION_PER_PAIR = 36_864
EMBEDDING_DIM = 768
# fangyuan/hotpotqa_extractive_compressor: max_position_embeddings = 512.
DEFAULT_MAX_SEQUENCE_LENGTH = 512
PADDING_MODES = ("none", "batch-max", "case-max")


def _token_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list of integers")
    return [require_int(item, f"{name}[{i}]", minimum=1) for i, item in enumerate(value)]


def _batch_padded_lengths(lengths: list[int], batch_size: int) -> list[int]:
    """Pad consecutive batches to each batch's own maximum, order preserved."""
    effective: list[int] = []
    for start in range(0, len(lengths), batch_size):
        group = lengths[start : start + batch_size]
        effective.extend([max(group)] * len(group))
    return effective


def calculate(row: dict[str, Any]) -> dict[str, Any]:
    query_tokens = require_int(row["query_tokens"], "query_tokens", minimum=1)
    candidate_tokens = _token_list(row["candidate_tokens"], "candidate_tokens")
    if not candidate_tokens:
        raise ValueError("candidate_tokens must contain at least one sentence")

    max_sequence_length = require_int(
        row.get("max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH),
        "max_sequence_length",
        minimum=1,
    )
    raw_lengths = [query_tokens, *candidate_tokens]
    clamped_lengths = [min(length, max_sequence_length) for length in raw_lengths]
    truncated_sequences = sum(
        1 for raw, clamped in zip(raw_lengths, clamped_lengths) if clamped < raw
    )

    padding_mode = row.get("padding_mode", "none")
    if padding_mode not in PADDING_MODES:
        raise ValueError(f"padding_mode must be one of {PADDING_MODES}")

    batch_size_value = row.get("batch_size")
    batch_size = (
        None
        if batch_size_value is None
        else require_int(batch_size_value, "batch_size", minimum=1)
    )
    if padding_mode == "batch-max" and batch_size is None:
        raise ValueError("padding_mode 'batch-max' requires batch_size")
    if padding_mode != "batch-max" and batch_size is not None:
        raise ValueError("batch_size is only meaningful with padding_mode 'batch-max'")

    padded_length = None
    if padding_mode == "none":
        effective_lengths = clamped_lengths
    elif padding_mode == "batch-max":
        # The query is encoded in a call of its own, so only the candidates are
        # batched together.
        effective_lengths = [
            clamped_lengths[0],
            *_batch_padded_lengths(clamped_lengths[1:], batch_size),
        ]
    else:
        padded_length = require_int(
            row.get("padded_length", max(clamped_lengths)),
            "padded_length",
            minimum=max(clamped_lengths),
        )
        effective_lengths = [padded_length] * len(clamped_lengths)

    sum_tokens = sum(effective_lengths)
    sum_squared_tokens = sum(length * length for length in effective_lengths)
    linear_flops = ENCODER_LINEAR_PER_TOKEN * sum_tokens
    attention_flops = ENCODER_ATTENTION_PER_PAIR * sum_squared_tokens
    dot_product_flops = 2 * EMBEDDING_DIM * len(candidate_tokens)
    return {
        "flops": linear_flops + attention_flops + dot_product_flops,
        "query_tokens": query_tokens,
        "candidate_tokens": candidate_tokens,
        "num_candidates": len(candidate_tokens),
        "max_sequence_length": max_sequence_length,
        "truncated_sequences": truncated_sequences,
        "padding_mode": padding_mode,
        "batch_size": batch_size,
        "padded_length": padded_length,
        "effective_sequence_lengths": effective_lengths,
        "padding_overhead_tokens": sum_tokens - sum(clamped_lengths),
        "linear_flops": linear_flops,
        "attention_flops": attention_flops,
        "dot_product_flops": dot_product_flops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate RECOMP-extractive major-matmul inference FLOPs."
    )
    add_batch_arguments(parser)
    parser.add_argument("--query-tokens", type=int)
    parser.add_argument("--candidate-tokens", type=int, nargs="+")
    parser.add_argument(
        "--padding-mode",
        choices=PADDING_MODES,
        default="none",
        help=(
            "'batch-max' models the runner's consecutive batches of candidates; "
            "'case-max' models the official one-case padded batch."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Candidate batch size; required by --padding-mode batch-max.",
    )
    parser.add_argument(
        "--padded-length",
        type=int,
        help="Optional actual padded batch length; otherwise the case maximum is used.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Truncation length applied before padding (checkpoint limit: 512).",
    )
    args = parser.parse_args()

    try:
        if args.input_jsonl:
            if args.query_tokens is not None or args.candidate_tokens is not None:
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
        if args.query_tokens is None or args.candidate_tokens is None:
            raise ValueError(
                "single-case mode requires --query-tokens and --candidate-tokens"
            )
        row: dict[str, Any] = {
            "query_tokens": args.query_tokens,
            "candidate_tokens": args.candidate_tokens,
            "padding_mode": args.padding_mode,
            "max_sequence_length": args.max_sequence_length,
        }
        if args.batch_size is not None:
            row["batch_size"] = args.batch_size
        if args.padded_length is not None:
            row["padded_length"] = args.padded_length
        print_json(result_record(METHOD, row, calculate(row)))
    except (KeyError, TypeError, ValueError) as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()
