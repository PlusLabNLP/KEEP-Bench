#!/usr/bin/env python3
"""Run the compression benchmark through the OpenAI Responses API (GPT-5.4 mini).

GPT-5.4 mini is one of the two proprietary compressors in the paper. It runs at
temperature 0 with reasoning off, on the same 27 tasks, prompts and
``compressed_context`` output key as ``run_compression.py``; only the transport
differs. Everything except the request and the response parsing comes from that
module: this script rebinds one function on it, ``process_one``, which is the only
place a request is issued. See ``install_seam``.

    27 tasks = 3 settings x 9 subsets x 3 budgets, 4102 rows per budget -> 12306 requests

OpenAI specifics
----------------
* Responses API, not chat completions. ``client.responses.create(input=...)`` accepts
  the role/content list that ``build_messages`` produces, system turn included, so the
  prompt needs no reshaping (Anthropic, by contrast, takes the system prompt as a separate
  top-level parameter).
* Reasoning is turned off with ``reasoning={"effort": "none"}``, the GPT counterpart of
  the ``without_thinking`` setting.
* The output cap is ``max_output_tokens``, not ``max_tokens``.
* ``store=False`` keeps inputs and outputs out of server-side response storage.
* SDK retries are disabled (``max_retries=0``) because this script owns the retry loop:
  a request that exhausts its retries writes the ``ERROR_COMPRESSED_CONTEXT`` sentinel, and
  ``--retry-errors`` re-requests exactly those rows later.

There is no context preflight: the hosted model's window is far above the benchmark's
longest prompt (22497 tokens).

Usage
-----
    export OPENAI_API_KEY=...
    python src/compression/run_compression_gpt.py --dry-run
    python src/compression/run_compression_gpt.py
    python src/compression/run_compression_gpt.py --setting conditional --budget 1x
    python src/compression/run_compression_gpt.py --retry-errors

Outputs are written to ``outputs/gpt/5-4-mini/without_thinking/<budget>/<task>.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import random
import sys
from pathlib import Path

import openai
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('run_compression', HERE.parent / 'compression' / 'run_compression.py')
rc = importlib.util.module_from_spec(_spec)
sys.modules['run_compression'] = rc
_spec.loader.exec_module(rc)

MODEL = 'gpt-5.4-mini'
OUTPUT_DIR = 'gpt/5-4-mini'
API_KEY_ENV = 'OPENAI_API_KEY'
DEFAULT_MAX_CONCURRENCY = 8          # hosted API rate limits
DEFAULT_MAX_RETRIES = 5
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_MAX_OUTPUT_TOKENS = 1024     # largest designed compression is 640 tokens


def extract_text(response) -> str:
    text = getattr(response, 'output_text', None)
    if text:
        return text.strip()
    chunks = []
    for item in getattr(response, 'output', None) or []:
        for content in getattr(item, 'content', None) or []:
            if getattr(content, 'type', None) == 'output_text':
                chunks.append(getattr(content, 'text', ''))
    return ''.join(chunks).strip()


async def process_one(idx, entry, client, semaphore, task, args, family, icl_examples):
    kwargs = {
        'model': args.served_model,
        'input': rc.build_messages(entry, task.k, task.prompt_family, icl_examples),
        'reasoning': {'effort': 'none'},          # the without_thinking arm
        'temperature': 0,
        'max_output_tokens': args.max_tokens,
        'store': False,
        'timeout': args.request_timeout,
    }
    async with semaphore:
        last = None
        for attempt in range(args.max_retries + 1):
            try:
                response = await client.responses.create(**kwargs)
                text = extract_text(response)
                if not text:
                    raise ValueError('empty response (status=%r)' % getattr(response, 'status', None))
                if getattr(response, 'status', None) == 'incomplete':
                    # Truncated at max_output_tokens. Keep the partial summary (a retry at
                    # temperature 0 would truncate identically) and log a warning.
                    print('WARN %s idx %d: incomplete (%r), kept %d chars'
                          % (task.name, idx, getattr(getattr(response, 'incomplete_details', None),
                                                    'reason', None), len(text)))
                out = dict(entry)
                out['compressed_context'] = text
                return idx, out
            # Most specific first: a 400/404/401 will never succeed on retry, so fail it now.
            except (openai.BadRequestError, openai.NotFoundError,
                    openai.AuthenticationError, openai.PermissionDeniedError) as error:
                last = error
                break
            except (openai.RateLimitError, openai.APIStatusError,
                    openai.APIConnectionError, openai.APITimeoutError, ValueError) as error:
                last = error
                if attempt < args.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 60) + random.random())
        print('Error in %s at index %d: %s: %s' % (task.name, idx, type(last).__name__, last))
        out = dict(entry)
        out['compressed_context'] = rc.ERROR_COMPRESSED_CONTEXT
        return idx, out


def install_seam():
    """Route run_compression's per-row request through the Responses API."""
    assert callable(rc.process_one), 'run_compression.process_one is missing'
    rc.process_one = process_one


async def probe(client, args):
    """One tiny live request: catches a bad key or a wrong model id before 12306 of them."""
    response = await client.responses.create(
        model=args.served_model, input=[{'role': 'user', 'content': 'Reply with the single word OK.'}],
        reasoning={'effort': 'none'}, temperature=0, max_output_tokens=16,
        store=False, timeout=args.request_timeout)
    text = extract_text(response)
    if not text:
        raise SystemExit('PREFLIGHT FAILED: the probe request returned no text (status=%r)'
                         % getattr(response, 'status', None))
    usage = getattr(response, 'usage', None)
    print('PREFLIGHT OK: %s is reachable, probe returned %r%s'
          % (args.served_model, text[:40],
             (' (input %s / output %s tokens)' % (usage.input_tokens, usage.output_tokens)) if usage else ''))


def parse_args():
    p = argparse.ArgumentParser(description='Context compression with GPT-5.4 mini through the OpenAI Responses API.')
    p.add_argument('--model', dest='served_model', default=MODEL)
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--api-key-env', default=API_KEY_ENV)
    p.add_argument('--base-url', default=None, help='Alternative endpoint, e.g. a gateway; default: the OpenAI API.')
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--max-concurrency', type=int, default=DEFAULT_MAX_CONCURRENCY)
    p.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument('--request-timeout', type=float, default=DEFAULT_REQUEST_TIMEOUT)
    p.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_OUTPUT_TOKENS,
                   help='Sent as max_output_tokens (the Responses API output cap).')
    p.add_argument('--setting', choices=['all', 'parallel', 'conditional', 'imbalanced'], default='all')
    p.add_argument('--budget', choices=['all', *rc.BUDGETS], default='all')
    p.add_argument('--hop', choices=['all', *rc.HOPS], default='all')
    p.add_argument('--domain', choices=['all', 'science', 'policy'], default='all')
    p.add_argument('--perspective', choices=['all', *rc.PERSPECTIVES], default='all')
    p.add_argument('--limit-per-task', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-preflight', action='store_true')
    args = p.parse_args()
    if args.overwrite and args.retry_errors:
        p.error('--overwrite and --retry-errors are mutually exclusive')
    args.family = 'gpt'
    args.enable_thinking = False
    if args.base_url is None:
        args.base_url = 'https://api.openai.com/v1'
    return args


async def main():
    args = parse_args()
    install_seam()
    tasks = [t for t in rc.all_tasks(args.data_root, args.output_root,
                                     args.output_dir, 'without_thinking')
             if rc.selected(t, args)]
    print('Selected %d task(s) for %s (openai responses, without_thinking).'
          % (len(tasks), args.served_model))
    missing = [str(t.input_file) for t in tasks if not t.input_file.exists()]
    if missing:
        raise SystemExit('Missing input file(s):\n  ' + '\n  '.join(sorted(set(missing))))

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit('No API key found: export %s=... first' % args.api_key_env)
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, max_retries=0)
    try:
        if not args.skip_preflight:
            await probe(client, args)
        if args.dry_run:
            for t in tasks:
                rc.print_task(t, args)
            return
        for t in tasks:
            # family only matters for locally served models (thinking / channel handling);
            # it is unused on this path, so None is passed.
            await rc.run_task(t, args, None, client)
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(main())
