#!/usr/bin/env python3
"""Run the compression benchmark through the Anthropic API (Claude Haiku 4.5).

Claude Haiku 4.5 is one of the two proprietary compressors in the paper. It runs at
temperature 0 without thinking, on the same 27 tasks, prompts and
``compressed_context`` output key as ``run_compression.py``; only the transport
differs. Everything except the request and the response parsing comes from that
module: this script rebinds one function on it, ``process_one``, which is the only
place a request is issued. See ``install_seam``.

    27 tasks = 3 settings x 9 subsets x 3 budgets, 4102 rows per budget -> 12306 requests

Anthropic specifics
-------------------
* The system prompt is a top-level parameter, not a message. ``build_messages`` from
  ``run_compression`` returns ``[system, *turns]`` for the OpenAI-style chat API, so the
  first element is split off here and passed as ``system=``.
* Thinking is off when the ``thinking`` parameter is omitted. Haiku 4.5 turns thinking on
  only with ``thinking={"type": "enabled", "budget_tokens": N}``, so this script does not
  send ``thinking`` at all. ``output_config={"effort": ...}`` is not sent either; Haiku 4.5
  rejects it.
* The default model id ``claude-haiku-4-5`` has no date suffix. Pass a dated snapshot id
  with ``--model`` (e.g. ``claude-haiku-4-5-20251001``) to pin a specific snapshot.
* The 200K context window is far above the benchmark's longest prompt (22497 tokens), so
  there is no context preflight.
* SDK retries are disabled (``max_retries=0``) because this script owns the retry loop:
  a request that exhausts its retries writes the ``ERROR_COMPRESSED_CONTEXT`` sentinel, and
  ``--retry-errors`` re-requests exactly those rows later.

Usage
-----
    export ANTHROPIC_API_KEY=...            # or: ant auth login
    python src/compression/run_compression_claude.py --dry-run
    python src/compression/run_compression_claude.py
    python src/compression/run_compression_claude.py --setting conditional --budget 1x
    python src/compression/run_compression_claude.py --retry-errors

Outputs are written to ``outputs/claude/haiku-4-5/without_thinking/<budget>/<task>.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import random
import sys
from pathlib import Path

import anthropic
from anthropic import AsyncAnthropic

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('run_compression', HERE.parent / 'compression' / 'run_compression.py')
rc = importlib.util.module_from_spec(_spec)
sys.modules['run_compression'] = rc
_spec.loader.exec_module(rc)

MODEL = 'claude-haiku-4-5'
OUTPUT_DIR = 'claude/haiku-4-5'
API_KEY_ENV = 'ANTHROPIC_API_KEY'
DEFAULT_MAX_CONCURRENCY = 8          # hosted API rate limits
DEFAULT_MAX_RETRIES = 5
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_MAX_TOKENS = 1024            # largest designed compression is 640 tokens


def split_system(messages):
    """[system, *turns] -> (system text, turns). Anthropic takes the system separately."""
    assert messages and messages[0]['role'] == 'system', ('first message is not a system turn', messages[0]['role'])
    return messages[0]['content'], messages[1:]


def extract_text(response) -> str:
    return ''.join(b.text for b in response.content if getattr(b, 'type', None) == 'text').strip()


async def process_one(idx, entry, client, semaphore, task, args, family, icl_examples):
    system, turns = split_system(
        rc.build_messages(entry, task.k, task.prompt_family, icl_examples))
    kwargs = {
        'model': args.served_model,
        'system': system,
        'messages': turns,
        'temperature': 0,
        'max_tokens': args.max_tokens,
        'timeout': args.request_timeout,
        # No `thinking` key: on Haiku 4.5 omitting it is what "no thinking" means.
    }
    async with semaphore:
        last = None
        for attempt in range(args.max_retries + 1):
            try:
                response = await client.messages.create(**kwargs)
                if response.stop_reason == 'refusal':
                    raise RuntimeError('refusal: %s' % getattr(response.stop_details, 'category', None))
                text = extract_text(response)
                if not text:
                    raise ValueError('empty response (stop_reason=%r)' % response.stop_reason)
                if response.stop_reason == 'max_tokens':
                    # Truncated at max_tokens. Keep the partial summary (a retry at
                    # temperature 0 would truncate identically) and log a warning.
                    print('WARN %s idx %d: hit max_tokens, kept %d chars' % (task.name, idx, len(text)))
                out = dict(entry)
                out['compressed_context'] = text
                return idx, out
            # Most specific first: a 400/404 will never succeed on retry, so fail it now.
            except (anthropic.BadRequestError, anthropic.NotFoundError,
                    anthropic.AuthenticationError, anthropic.PermissionDeniedError) as error:
                last = error
                break
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    anthropic.APIConnectionError, anthropic.APITimeoutError,
                    RuntimeError, ValueError) as error:
                last = error
                if attempt < args.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 60) + random.random())
        print('Error in %s at index %d: %s: %s' % (task.name, idx, type(last).__name__, last))
        out = dict(entry)
        out['compressed_context'] = rc.ERROR_COMPRESSED_CONTEXT
        return idx, out


def install_seam():
    """Route run_compression's per-row request through the Anthropic client."""
    assert callable(rc.process_one), 'run_compression.process_one is missing'
    rc.process_one = process_one


async def probe(client, args):
    """One tiny live request: catches a bad key or a wrong model id before 12306 of them."""
    response = await client.messages.create(
        model=args.served_model, max_tokens=16, temperature=0,
        timeout=args.request_timeout,
        messages=[{'role': 'user', 'content': 'Reply with the single word OK.'}])
    text = extract_text(response)
    if not text:
        raise SystemExit('PREFLIGHT FAILED: the probe request returned no text (stop_reason=%r)' % response.stop_reason)
    print('PREFLIGHT OK: %s is reachable, probe returned %r (input %d / output %d tokens)'
          % (args.served_model, text[:40],
             response.usage.input_tokens, response.usage.output_tokens))


def parse_args():
    p = argparse.ArgumentParser(description='Context compression with Claude Haiku 4.5 through the Anthropic API.')
    p.add_argument('--model', dest='served_model', default=MODEL,
                   help='Default %s (no date suffix). Pass a dated id such as claude-haiku-4-5-20251001 to pin a snapshot.' % MODEL)
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--api-key-env', default=API_KEY_ENV)
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--max-concurrency', type=int, default=DEFAULT_MAX_CONCURRENCY)
    p.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument('--request-timeout', type=float, default=DEFAULT_REQUEST_TIMEOUT)
    p.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS)
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
    # Attributes read by run_compression.print_task and run_task.
    args.family = 'claude'
    args.base_url = 'https://api.anthropic.com'
    args.enable_thinking = False
    return args


async def main():
    args = parse_args()
    install_seam()
    tasks = [t for t in rc.all_tasks(args.data_root, args.output_root,
                                     args.output_dir, 'without_thinking')
             if rc.selected(t, args)]
    print('Selected %d task(s) for %s (anthropic, without_thinking).' % (len(tasks), args.served_model))
    missing = [str(t.input_file) for t in tasks if not t.input_file.exists()]
    if missing:
        raise SystemExit('Missing input file(s):\n  ' + '\n  '.join(sorted(set(missing))))

    api_key = os.getenv(args.api_key_env)
    # Without an explicit api_key the SDK also reads ANTHROPIC_AUTH_TOKEN and the
    # `ant auth login` profile, so a missing environment variable is left to the SDK.
    client = AsyncAnthropic(max_retries=0, **({'api_key': api_key} if api_key else {}))
    try:
        if not args.skip_preflight:
            await probe(client, args)
        if args.dry_run:
            for t in tasks:
                rc.print_task(t, args)
            return
        for t in tasks:
            # family only matters for locally served models (thinking / channel handling);
            # it is unused on this path, so None is passed rather than another family's spec.
            await rc.run_task(t, args, None, client)
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(main())
