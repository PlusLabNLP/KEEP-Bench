#!/usr/bin/env python3
r"""Major-matmul inference FLOPs for the 11 ICL compressors, with optional prefix caching.

The 11 compressors differ only in their coefficients and share two forward shapes:

    dense           F = L.T + A.A(T) + H.Z                    Qwen3, Llama-3
    sliding+global  F = L.T + A_loc.A_W(T) + A_glb.A(T) + H.Z  Gemma-4

with one multiply-add = 2 FLOPs, a KV cache, and efficient causal attention:

    T = P + G - 1          processed Transformer positions
    A(T) = T(T+1)/2        useful causal query-key pairs
    A_W(T) = sum min(i,W)  the same under a sliding window of W
    Z                      vocabulary-head positions, defaults to G

P is the complete prompt length under the model's own tokenizer and chat template, G the
actual generated token-ID count. Norms, RoPE, activations, softmax, sampling, embedding
lookups and communication are excluded, as are Gemma's vision/audio towers.

Coefficients are computed from each model's ``config.json``. Two Gemma-4 details matter:
``global_head_dim`` gives the full-attention layers a 2x head dim, so the global term is
twice what the head count alone suggests, and ``attention_k_eq_v`` removes the separate
global V projection.

Prefix caching
--------------
``--prefix-tokens p`` reports the cost when every request in the batch starts with the same
p tokens, as when one setting's ICL block is shared across its cases:

    F_cached(row; p) = F_full(row) - F_prefix(p, G=1, Z=0)

G=1 makes the helper's processed length exactly p and Z=0 drops its vocabulary projection,
so the subtraction removes only the work that built the reusable prefix KV. Because
``causal_pairs`` is a partial sum, attention *from* the new tokens *to* the cached prefix
is still counted. The same holds under a sliding window, since ``sliding_causal_pairs`` is
also a partial sum.

A batch run reports both the warm-cache steady state and the total with one warm-up
amortized over the batch. Validity: every cached token, special token and position must be
identical, so anything case-specific before the shared block ends the reusable prefix.

Usage
-----
    python src/computation/llm_flops.py --list-models
    python src/computation/llm_flops.py --model qwen3-8b --prompt-tokens 16771 \
        --generated-tokens 200
    python src/computation/llm_flops.py --model qwen3-8b --prompt-tokens 16771 \
        --generated-tokens 200 --prefix-tokens 13716
    python src/computation/llm_flops.py --model gemma4-31b --input-jsonl cases.jsonl \
        --prefix-tokens 13457 --output-jsonl per_case.jsonl

A JSONL row is ``{"prompt_tokens": P, "generated_tokens": G, "logit_positions": Z?}``.
Importable: ``MODELS``, ``calculator_for(alias)``, ``prefix_cached_calculate(row, ...)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from flops_cli import (
    add_batch_arguments,
    add_generation_arguments,
    causal_pairs,
    fail,
    generation_lengths,
    print_json,
    read_jsonl,
    require_int,
    result_record,
    run_batch,
    sliding_causal_pairs,
)

JsonDict = dict[str, Any]

# --------------------------------------------------------------------------- #
# Coefficients, computed from each model's config.json.
#
#   linear   per processed position: Q/K/V/O + MLP, plus for Gemma-4 the KV-shared layers,
#            double-wide MLPs and per-layer-embedding projections
#   attn     per causal query-key pair: 4 . head_dim . heads . layers
#   lm_head  per vocabulary-head position: 2 . hidden . vocab
#   window   sliding-window width; None means every layer is full attention
# --------------------------------------------------------------------------- #

MODELS: dict[str, dict[str, Any]] = {
    'qwen3-4b': dict(method='Qwen3-4B', linear=7_266_631_680, attn=589_824,
                     lm_head=777_912_320),
    'qwen3-8b': dict(method='Qwen3-8B', linear=13_891_534_848, attn=589_824,
                     lm_head=1_244_659_712),
    'qwen3-14b': dict(method='Qwen3-14B', linear=26_424_115_200, attn=819_200,
                      lm_head=1_555_824_640),
    'qwen3-32b': dict(method='Qwen3-32B', linear=62_411_243_520, attn=2_097_152,
                      lm_head=1_555_824_640),
    'llama3.2-1b': dict(method='Llama-3.2-1B-Instruct', linear=1_946_157_056,
                        attn=131_072, lm_head=525_336_576),
    'llama3.2-3b': dict(method='Llama-3.2-3B-Instruct', linear=5_637_144_576,
                        attn=344_064, lm_head=788_004_864),
    'llama3.1-8b': dict(method='Llama-3.1-8B-Instruct', linear=13_958_643_712,
                        attn=524_288, lm_head=1_050_673_152),
    # Gemma-4: `attn` is the sliding-window term, `attn_global` the full-attention one.
    'gemma4-e2b': dict(method='Gemma-4-E2B-it', linear=3_753_639_936, attn=229_376,
                       attn_global=114_688, lm_head=805_306_368, window=512),
    'gemma4-e4b': dict(method='Gemma-4-E4B-it', linear=7_945_584_640, attn=286_720,
                       attn_global=114_688, lm_head=1_342_177_280, window=512),
    'gemma4-12b': dict(method='Gemma-4-12B-it', linear=21_799_895_040, attn=655_360,
                       attn_global=262_144, lm_head=2_013_265_920, window=1024),
    'gemma4-31b': dict(method='Gemma-4-31B-it', linear=58_573_455_360, attn=1_638_400,
                       attn_global=655_360, lm_head=2_818_572_288, window=1024),
}


def calculate(row: JsonDict, spec: JsonDict) -> JsonDict:
    """FLOPs for one request under *spec*. Dense or sliding+global, by spec['window']."""
    prompt, generated, logits, processed = generation_lengths(row)
    window = spec.get('window')
    linear_flops = spec['linear'] * processed
    lm_head_flops = spec['lm_head'] * logits
    out: JsonDict = {
        'prompt_tokens': prompt,
        'generated_tokens': generated,
        'processed_tokens': processed,
        'logit_positions': logits,
    }
    # The *_flops keys must partition the total exactly: prefix_cached_calculate() sums
    # them as an invariant. So the sliding case reports local and global separately and
    # never also their sum.
    if window is None:
        pairs = causal_pairs(processed)
        attention = {'attention_pairs': pairs, 'attention_flops': spec['attn'] * pairs}
    else:
        local_pairs = sliding_causal_pairs(processed, window)
        global_pairs = causal_pairs(processed)
        attention = {
            'sliding_window': window,
            'local_attention_pairs': local_pairs,
            'global_attention_pairs': global_pairs,
            'local_attention_flops': spec['attn'] * local_pairs,
            'global_attention_flops': spec['attn_global'] * global_pairs,
        }
    attention_flops = sum(v for k, v in attention.items() if k.endswith('_flops'))
    out['flops'] = linear_flops + attention_flops + lm_head_flops
    out['linear_flops'] = linear_flops
    out.update(attention)
    out['lm_head_flops'] = lm_head_flops
    return out


def calculator_for(alias: str) -> Callable[[JsonDict], JsonDict]:
    if alias not in MODELS:
        raise ValueError('unknown model %r; try --list-models' % alias)
    spec = MODELS[alias]
    return lambda row: calculate(row, spec)


# --------------------------------------------------------------------------- #
# Prefix caching
# --------------------------------------------------------------------------- #

def prefix_cached_calculate(row: JsonDict, *, prefix_tokens: int,
                           base_calculator: Callable[[JsonDict], JsonDict]) -> JsonDict:
    """Steady-state FLOPs once one exact common prefix of *prefix_tokens* is cached."""
    prompt, generated, logits, processed = generation_lengths(row)
    prefix_tokens = require_int(prefix_tokens, 'prefix_tokens', minimum=1)
    if prefix_tokens > prompt:
        raise ValueError('prefix_tokens (%d) must be <= prompt_tokens (%d)'
                         % (prefix_tokens, prompt))

    full = base_calculator(row)
    # G=1 -> processed length is exactly p; Z=0 -> no vocabulary projection.
    prefix = base_calculator({'prompt_tokens': prefix_tokens, 'generated_tokens': 1,
                              'logit_positions': 0})
    full_flops = require_int(full['flops'], 'full_flops', minimum=0)
    prefix_flops = require_int(prefix['flops'], 'prefix_prefill_flops', minimum=0)
    cached = full_flops - prefix_flops
    if cached < 0:
        raise ValueError('prefix prefill FLOPs exceed full-request FLOPs; check prefix length')

    deltas = {k: full[k] - prefix[k] for k in full
              if k.endswith(('_flops', '_pairs')) and k != 'flops' and k in prefix}
    component_total = sum(v for k, v in deltas.items() if k.endswith('_flops'))
    if deltas and component_total != cached:
        raise ValueError('cached component FLOPs do not sum to cached total FLOPs')

    return {
        'flops': cached,                       # warm cache, steady state
        'prompt_tokens': prompt,
        'generated_tokens': generated,
        'processed_tokens': processed,
        'logit_positions': logits,
        'prefix_tokens': prefix_tokens,
        'processed_tokens_after_cached_prefix': processed - prefix_tokens,
        'full_flops_without_prefix_cache': full_flops,
        'cached_prefix_prefill_flops': prefix_flops,
        'cached_component_deltas': deltas,
        'full_flops_details': full,
        'cached_prefix_prefill_details': prefix,
    }


def run_prefix_cached_batch(*, method: str, base_calculator, input_path: str,
                            output_path: str | None, prefix_tokens: int) -> JsonDict:
    """One batch sharing one prefix: warm-cache steady state and one amortized warm-up."""
    if output_path == '-':
        raise ValueError("--output-jsonl cannot be '-' because stdout is reserved for summary")
    handle = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        handle = Path(output_path).open('w', encoding='utf-8')

    count = total_full = total_cached = 0
    warmup: int | None = None
    try:
        for line_number, row in read_jsonl(input_path):
            try:
                d = prefix_cached_calculate(row, prefix_tokens=prefix_tokens,
                                            base_calculator=base_calculator)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('%s:%d: %s' % (input_path, line_number, exc)) from exc
            if warmup is None:
                warmup = d['cached_prefix_prefill_flops']
            elif warmup != d['cached_prefix_prefill_flops']:
                raise ValueError('fixed prefix produced inconsistent FLOPs')
            count += 1
            total_full += d['full_flops_without_prefix_cache']
            total_cached += d['flops']
            if handle is not None:
                record = dict(row)
                record.update(method='%s (prefix-cached)' % method, flops=d['flops'],
                              tflops=d['flops'] / 1e12,
                              flops_details={k: v for k, v in d.items() if k != 'flops'})
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    finally:
        if handle is not None:
            handle.close()
    if count == 0 or warmup is None:
        raise ValueError('%s: no JSON objects found' % input_path)

    amortized = total_cached + warmup
    savings = total_full - amortized
    return {
        'method': '%s (prefix-cached)' % method,
        'num_cases': count,
        'prefix_tokens': prefix_tokens,
        'prefix_cache_warmup_flops': warmup,
        'prefix_cache_warmup_tflops': warmup / 1e12,
        'total_uncached_flops': total_full,
        'total_uncached_tflops': total_full / 1e12,
        'total_steady_state_cached_flops': total_cached,
        'total_steady_state_cached_tflops': total_cached / 1e12,
        'total_amortized_cached_flops': amortized,
        'total_amortized_cached_tflops': amortized / 1e12,
        'total_prefix_cache_savings_after_warmup_flops': savings,
        'total_prefix_cache_savings_after_warmup_tflops': savings / 1e12,
        'prefix_cache_reduction_after_warmup': savings / total_full,
        'average_uncached_tflops_per_case': total_full / count / 1e12,
        'average_steady_state_cached_tflops_per_case': total_cached / count / 1e12,
        'average_amortized_cached_tflops_per_case': amortized / count / 1e12,
        'per_case_output': output_path,
    }


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', help='alias from --list-models')
    p.add_argument('--list-models', action='store_true')
    add_batch_arguments(p)
    add_generation_arguments(p)
    p.add_argument('--prefix-tokens', type=int,
                   help='exact longest common token prefix shared by every request')
    args = p.parse_args()

    if args.list_models:
        print('%-13s %-24s %14s %10s %13s %7s'
              % ('alias', 'method', 'linear/token', 'attn/pair', 'lm_head/pos', 'window'))
        for alias, s in MODELS.items():
            print('%-13s %-24s %14d %10d %13d %7s'
                  % (alias, s['method'], s['linear'], s['attn'], s['lm_head'],
                     s.get('window', '-')))
        return

    try:
        if not args.model:
            raise ValueError('--model is required (see --list-models)')
        spec = MODELS.get(args.model)
        if spec is None:
            raise ValueError('unknown model %r; try --list-models' % args.model)
        calc = calculator_for(args.model)

        if args.input_jsonl:
            if args.prompt_tokens is not None or args.generated_tokens is not None:
                raise ValueError('use either --input-jsonl or single-case token arguments')
            if args.prefix_tokens is not None:
                print_json(run_prefix_cached_batch(
                    method=spec['method'], base_calculator=calc,
                    input_path=args.input_jsonl, output_path=args.output_jsonl,
                    prefix_tokens=args.prefix_tokens))
            else:
                print_json(run_batch(method=spec['method'], input_path=args.input_jsonl,
                                     output_path=args.output_jsonl, calculator=calc))
            return

        if args.output_jsonl:
            raise ValueError('--output-jsonl requires --input-jsonl')
        if args.prompt_tokens is None or args.generated_tokens is None:
            raise ValueError('single-case mode requires --prompt-tokens and --generated-tokens')
        row: JsonDict = {'prompt_tokens': args.prompt_tokens,
                         'generated_tokens': args.generated_tokens}
        if args.logit_positions is not None:
            row['logit_positions'] = args.logit_positions

        if args.prefix_tokens is None:
            print_json(result_record(spec['method'], row, calc(row)))
            return
        d = prefix_cached_calculate(row, prefix_tokens=args.prefix_tokens,
                                   base_calculator=calc)
        out = result_record('%s (prefix-cached)' % spec['method'], row, d)
        out['single_case_flops_including_cache_warmup'] = (
            d['flops'] + d['cached_prefix_prefill_flops'])
        out['single_case_tflops_including_cache_warmup'] = (
            out['single_case_flops_including_cache_warmup'] / 1e12)
        print_json(out)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        fail(str(exc))


if __name__ == '__main__':
    main()
