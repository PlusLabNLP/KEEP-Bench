#!/usr/bin/env python3
"""Shared, dependency-free helpers for the KEEP-Bench FLOPs calculators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


JsonDict = dict[str, Any]


def require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    """Return *value* as an int, rejecting booleans and non-integral values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def causal_pairs(length: int) -> int:
    """Number of useful causal query-key pairs for a sequence of *length*."""
    length = require_int(length, "length", minimum=0)
    return length * (length + 1) // 2


def sliding_causal_pairs(length: int, window: int) -> int:
    """Useful causal pairs when each query attends to at most *window* keys."""
    length = require_int(length, "length", minimum=0)
    window = require_int(window, "window", minimum=1)
    if length <= window:
        return causal_pairs(length)
    return window * length - window * (window - 1) // 2


def result_record(method: str, source: JsonDict, details: JsonDict) -> JsonDict:
    """Merge an input record with a calculator result."""
    flops = require_int(details["flops"], "flops", minimum=0)
    output = dict(source)
    output.update(
        {
            "method": method,
            "flops": flops,
            "tflops": flops / 1e12,
            "flops_details": {k: v for k, v in details.items() if k != "flops"},
        }
    )
    return output


def read_jsonl(path: str) -> Iterable[tuple[int, JsonDict]]:
    """Yield (line_number, object) from a UTF-8 JSONL file or stdin ('-')."""
    handle = sys.stdin if path == "-" else Path(path).open("r", encoding="utf-8")
    should_close = handle is not sys.stdin
    try:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            yield line_number, value
    finally:
        if should_close:
            handle.close()


def print_json(value: JsonDict) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-jsonl",
        metavar="PATH",
        help="Read one case per JSONL line. Use '-' for stdin.",
    )
    parser.add_argument(
        "--output-jsonl",
        metavar="PATH",
        help="Optional enriched per-case JSONL output. Existing files are overwritten.",
    )


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        help="Complete model-tokenized prompt length P, including chat template and special tokens.",
    )
    parser.add_argument(
        "--generated-tokens",
        type=int,
        help="Actual generated token-ID count G, including EOS/control tokens returned by the backend.",
    )
    parser.add_argument(
        "--logit-positions",
        type=int,
        help="Vocabulary-head positions Z. Defaults to G for efficient generation.",
    )


def generation_lengths(row: JsonDict) -> tuple[int, int, int, int]:
    """Resolve P, G, Z, and processed Transformer positions T from a record."""
    prompt_tokens = require_int(row["prompt_tokens"], "prompt_tokens", minimum=1)
    generated_tokens = require_int(row["generated_tokens"], "generated_tokens", minimum=1)
    logit_positions = require_int(
        row.get("logit_positions", generated_tokens),
        "logit_positions",
        minimum=0,
    )
    processed_tokens = prompt_tokens + generated_tokens - 1
    return prompt_tokens, generated_tokens, logit_positions, processed_tokens


def run_generation_cli(
    *,
    method: str,
    description: str,
    calculator: Callable[[JsonDict], JsonDict],
) -> None:
    """CLI shared by decoder-only generation calculators."""
    parser = argparse.ArgumentParser(description=description)
    add_batch_arguments(parser)
    add_generation_arguments(parser)
    args = parser.parse_args()

    try:
        if args.input_jsonl:
            if args.prompt_tokens is not None or args.generated_tokens is not None:
                raise ValueError(
                    "use either --input-jsonl or single-case token arguments, not both"
                )
            summary = run_batch(
                method=method,
                input_path=args.input_jsonl,
                output_path=args.output_jsonl,
                calculator=calculator,
            )
            print_json(summary)
            return

        if args.output_jsonl:
            raise ValueError("--output-jsonl requires --input-jsonl")
        if args.prompt_tokens is None or args.generated_tokens is None:
            raise ValueError(
                "single-case mode requires --prompt-tokens and --generated-tokens"
            )
        row: JsonDict = {
            "prompt_tokens": args.prompt_tokens,
            "generated_tokens": args.generated_tokens,
        }
        if args.logit_positions is not None:
            row["logit_positions"] = args.logit_positions
        print_json(result_record(method, row, calculator(row)))
    except (KeyError, TypeError, ValueError) as exc:
        fail(str(exc))


def run_batch(
    *,
    method: str,
    input_path: str,
    output_path: str | None,
    calculator: Callable[[JsonDict], JsonDict],
) -> JsonDict:
    """Run *calculator* over JSONL and return an aggregate summary."""
    if output_path == "-":
        raise ValueError("--output-jsonl cannot be '-' because stdout is reserved for the summary")
    if output_path:
        output_handle = Path(output_path).open("w", encoding="utf-8")
    else:
        output_handle = None

    count = 0
    total_flops = 0
    try:
        for line_number, row in read_jsonl(input_path):
            try:
                details = calculator(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            enriched = result_record(method, row, details)
            count += 1
            total_flops += enriched["flops"]
            if output_handle is not None:
                output_handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()

    if count == 0:
        raise ValueError(f"{input_path}: no JSON objects found")

    return {
        "method": method,
        "num_cases": count,
        "total_flops": total_flops,
        "average_flops_per_case": total_flops / count,
        "average_tflops_per_case": total_flops / count / 1e12,
        "per_case_output": output_path,
    }


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)