#!/usr/bin/env python3
"""Estimate RECOMP-abstractive (T5-large) major-matmul inference FLOPs.

Let n be the encoder-input length, r the number of decoder steps
(`len(outputs.scores)`), and b the beam width. With KV-cached decoding:

    F_encoder = 603,979,776 n + 98,304 n^2
    F_decoder = b [770,441,216 r
                   + 98,304 (r(r+1)/2 + nr)]
    F_cross_KV = k 100,663,296 n

where k=b for standard beam-expanded execution and k=1 only if cross-attention
K/V projections are explicitly shared across beams.

Sources:
https://huggingface.co/fangyuan/nq_abstractive_compressor/blob/main/config.json
https://github.com/carriex/recomp
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


METHOD = "RECOMP-abstractive"
ENCODER_LINEAR_PER_TOKEN = 603_979_776
ENCODER_ATTENTION_PER_PAIR = 98_304
DECODER_LINEAR_AND_LM_PER_STEP = 770_441_216
DECODER_ATTENTION_PER_PAIR = 98_304
CROSS_KV_PER_ENCODER_TOKEN = 100_663_296


def calculate(row: dict[str, Any]) -> dict[str, Any]:
    encoder_tokens = require_int(row["encoder_tokens"], "encoder_tokens", minimum=1)
    decoder_steps = require_int(row["decoder_steps"], "decoder_steps", minimum=1)
    beam_width = require_int(row["beam_width"], "beam_width", minimum=1)
    share_cross_kv = row.get("share_cross_kv_across_beams", False)
    if not isinstance(share_cross_kv, bool):
        raise ValueError("share_cross_kv_across_beams must be true or false")

    decoder_pairs = causal_pairs(decoder_steps)
    encoder_linear_flops = ENCODER_LINEAR_PER_TOKEN * encoder_tokens
    encoder_attention_flops = ENCODER_ATTENTION_PER_PAIR * encoder_tokens**2
    decoder_linear_flops = (
        beam_width * DECODER_LINEAR_AND_LM_PER_STEP * decoder_steps
    )
    decoder_self_attention_flops = (
        beam_width * DECODER_ATTENTION_PER_PAIR * decoder_pairs
    )
    cross_attention_flops = (
        beam_width
        * DECODER_ATTENTION_PER_PAIR
        * encoder_tokens
        * decoder_steps
    )
    cross_kv_multiplier = 1 if share_cross_kv else beam_width
    cross_kv_flops = (
        cross_kv_multiplier * CROSS_KV_PER_ENCODER_TOKEN * encoder_tokens
    )
    return {
        "flops": (
            encoder_linear_flops
            + encoder_attention_flops
            + decoder_linear_flops
            + decoder_self_attention_flops
            + cross_attention_flops
            + cross_kv_flops
        ),
        "encoder_tokens": encoder_tokens,
        "decoder_steps": decoder_steps,
        "beam_width": beam_width,
        "share_cross_kv_across_beams": share_cross_kv,
        "decoder_causal_attention_pairs": decoder_pairs,
        "encoder_linear_flops": encoder_linear_flops,
        "encoder_attention_flops": encoder_attention_flops,
        "decoder_linear_and_lm_head_flops": decoder_linear_flops,
        "decoder_self_attention_flops": decoder_self_attention_flops,
        "cross_attention_flops": cross_attention_flops,
        "cross_kv_projection_flops": cross_kv_flops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate RECOMP-abstractive major-matmul inference FLOPs."
    )
    add_batch_arguments(parser)
    parser.add_argument("--encoder-tokens", type=int)
    parser.add_argument("--decoder-steps", type=int)
    parser.add_argument("--beam-width", type=int)
    parser.add_argument(
        "--share-cross-kv-across-beams",
        action="store_true",
        help="Use only if the backend computes cross-attention K/V once and shares it.",
    )
    args = parser.parse_args()

    try:
        if args.input_jsonl:
            if any(
                value is not None
                for value in (args.encoder_tokens, args.decoder_steps, args.beam_width)
            ):
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
        if any(
            value is None
            for value in (args.encoder_tokens, args.decoder_steps, args.beam_width)
        ):
            raise ValueError(
                "single-case mode requires --encoder-tokens, --decoder-steps, "
                "and --beam-width"
            )
        row = {
            "encoder_tokens": args.encoder_tokens,
            "decoder_steps": args.decoder_steps,
            "beam_width": args.beam_width,
            "share_cross_kv_across_beams": args.share_cross_kv_across_beams,
        }
        print_json(result_record(METHOD, row, calculate(row)))
    except (KeyError, TypeError, ValueError) as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()