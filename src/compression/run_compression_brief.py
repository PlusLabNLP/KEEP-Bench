#!/usr/bin/env python3
r"""Run the compression benchmark through BRIEF-Pro, served by vLLM.

Same 27 tasks, same ``compressed_context`` output key and same output layout as
``run_compression.py``; only the request differs. Everything except the request and the
response parsing is reused from that module: this script rebinds exactly one function on
it, ``process_one`` (see ``install_seam``).

    27 tasks = 3 settings x 9 subsets x 3 budgets, 4102 rows each -> 12306 requests

How BRIEF-Pro differs from the ICL models
-----------------------------------------
* It is a dedicated compressor, not an instruction-following LLM. It was trained on one
  prompt shape and ignores a system turn, so this script sends neither the benchmark's
  system prompt nor the ICL examples. ``run_task`` still loads the examples (the shot-count
  assertion lives there) and ``process_one`` ignores them. The compression is therefore
  zero-shot, unlike every ICL model.
* The sentence budget is passed in two reserved tokens, not in prose:
  ``... where K = <|reserved_special_token_100|>{k}<|reserved_special_token_101|>``.
  ``k`` comes from ``task.k``, so 0.5x/1x/2x map to the same per-hop sentence counts as the
  rest of the benchmark (``PARALLEL_K`` etc. in ``run_compression``).
* The chat template is applied here, not by the server, because the control tokens have
  to appear verbatim inside the user turn. The request therefore goes to the completions
  endpoint with a raw ``prompt`` string, not to the chat completions endpoint.
* ``date_string`` is pinned. The Llama-3 template inserts the current date into the
  system block, which would make the prompt, and so the output, depend on the day of the
  run. It is pinned to the Llama-3.1 default so reruns are comparable.

Relation to the official script
-------------------------------
The official reference is ``BRIEF-Pro/eval/vllm_compression_longbench_testAll_UserControl.py``
in https://github.com/JasonForJoy/BRIEF (the model card has no usage example). The control
tokens, the instruction wording and the four ``###`` markers match it exactly, including
the single space in ``question.\n ###``.

Differences:

1. The assistant turn. The official script appends
   ``"<|start_header_id|>assistant<|end_header_id|>"`` by hand after
   ``apply_chat_template(tokenize=False)``, which omits the ``\n\n`` the template adds.
   Here ``add_generation_prompt=True`` builds that turn, and no post-processing is needed.
2. The date. The tokenizer's template calls ``strftime_now``, so the official script puts
   the day of the run into the system block. ``date_string`` is pinned to the Llama-3.1
   default (``26 Jul 2024``) instead.
3. Decoding. The official script samples with ``temperature=0.01, top_p=0.95``. The rest of
   this benchmark is greedy, so the default here is ``temperature=0``; pass
   ``--temperature 0.01 --top-p 0.95`` to match the official setting.
4. K. The official script hardcodes ``control_length = 20``. Here ``k`` comes from
   ``task.k``, so 0.5x/1x/2x use the benchmark's per-hop counts.

``--prompt-variant simple`` (the default, and the only instruction in the official script)
is used for all results.

Usage
-----
    # 1. serve the model
    bash src/compression/serve_brief_pro.sh

    # 2. compress
    python src/compression/run_compression_brief.py --dry-run
    python src/compression/run_compression_brief.py
    python src/compression/run_compression_brief.py --budget 1x
    python src/compression/run_compression_brief.py --retry-errors

Outputs are written to ``outputs/brief-pro/3b/without_thinking/<budget>/<task>.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('run_compression', HERE.parent / 'compression' / 'run_compression.py')
rc = importlib.util.module_from_spec(_spec)
sys.modules['run_compression'] = rc
_spec.loader.exec_module(rc)

MODEL = 'brief-pro'                  # the name --served-model-name advertises
OUTPUT_DIR = 'brief-pro/3b'          # Llama-3.2-3B backbone, hence 3b
DEFAULT_BASE_URL = 'http://localhost:8000/v1'
DEFAULT_TOKENIZER = 'uclanlp/brief-pro'
DEFAULT_MAX_CONCURRENCY = 64         # the server queues what it cannot batch
DEFAULT_MAX_RETRIES = 5
DEFAULT_REQUEST_TIMEOUT = 600.0
DEFAULT_MAX_TOKENS = 1024            # largest designed compression is 640 tokens
PINNED_DATE = '26 Jul 2024'          # keep the templated system block reproducible

# BRIEF-Pro's trained instruction. The disambiguation variant names the ambiguity that the
# parallel setting tests, so it is not a like-for-like baseline against the other
# compressors. The paper's results use the simple one.
INSTRUCTION = {
    'simple': (
        'Write a high-quality summary of the provided documents with respect to the '
        'question.'
    ),
    'disambiguation': (
        'Write a high-quality summary of the provided documents with respect to the '
        'question.\n'
        'The question may contain potentially ambiguous entities, and within the documents '
        'some entities mentioned in the question may refer to multiple different referents.\n'
        'Preserve all such possible interpretations and supporting evidence in the summary.\n'
        'Include all relevant evidence for different possible referents, preserve '
        'conflicting evidence when present, and prioritize and front-load the most critical '
        'points while remaining concise.'
    ),
}

_TOKENIZER = None


def tokenizer(args):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer)
        for token in ('<|reserved_special_token_100|>', '<|reserved_special_token_101|>'):
            if _TOKENIZER.convert_tokens_to_ids(token) is None:
                raise SystemExit('The tokenizer has no control token %s: %s is not the BRIEF-Pro tokenizer'
                                 % (token, args.tokenizer))
    return _TOKENIZER


def build_prompt(entry, k: int, args) -> str:
    """The full raw prompt, chat template included, control tokens carrying k."""
    control = ('\nSummarize the documents relevant to the question in K sentences, '
               'where K = <|reserved_special_token_100|>%d<|reserved_special_token_101|>' % k)
    content = (
        # The space after the first \n matches the official UserControl script
        # ("question.\n ###"). Only the first ### is preceded by a space.
        '%s\n '
        '### This is the question: %s\n'
        '### These are the documents:\n%s\n'
        '### This is the summary:%s'
        % (INSTRUCTION[args.prompt_variant], entry.get('question', ''),
           entry.get('context', ''), control)
    )
    return tokenizer(args).apply_chat_template(
        [{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, date_string=PINNED_DATE)


async def process_one(idx, entry, client, semaphore, task, args, family, icl_examples):
    # icl_examples is deliberately unused: BRIEF-Pro is zero-shot by construction.
    kwargs = {
        'model': args.served_model,
        'prompt': build_prompt(entry, task.k, args),
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'timeout': args.request_timeout,
    }
    if args.top_p is not None:
        kwargs['top_p'] = args.top_p
    async with semaphore:
        for attempt in range(args.max_retries + 1):
            try:
                response = await client.completions.create(**kwargs)
                text = (response.choices[0].text or '').strip()
                if not text:
                    raise ValueError('empty completion (finish_reason=%r)'
                                     % response.choices[0].finish_reason)
                if response.choices[0].finish_reason == 'length':
                    # Truncated at max_tokens: a half-written summary. Keep it (greedy
                    # decoding would truncate identically) but say so -- a silent truncation
                    # here becomes a silently short compressed_context.
                    print('WARN %s idx %d: hit max_tokens, kept %d chars'
                          % (task.name, idx, len(text)))
                out = dict(entry)
                out['compressed_context'] = text
                return idx, out
            except Exception as error:                 # noqa: BLE001 -- recorded, not swallowed
                if attempt == args.max_retries:
                    print('Error in %s at index %d: %s: %s'
                          % (task.name, idx, type(error).__name__, error))
                    out = dict(entry)
                    out['compressed_context'] = rc.ERROR_COMPRESSED_CONTEXT
                    return idx, out
                await asyncio.sleep(min(2 ** attempt, 30) + random.random())


def install_seam():
    """Route run_compression's per-row request through the completions endpoint."""
    assert callable(rc.process_one), 'run_compression.process_one does not exist'
    rc.process_one = process_one


async def probe(client, args):
    """Catch a server that is down, or serving something else, before 12306 requests."""
    served = {m.id for m in (await client.models.list()).data}
    if args.served_model not in served:
        raise SystemExit('PREFLIGHT FAILED: %s is not serving %r (serving %s)\n'
                         'Start the server: bash src/compression/serve_brief_pro.sh'
                         % (args.base_url, args.served_model, sorted(served) or 'nothing'))
    response = await client.completions.create(
        model=args.served_model,
        prompt=build_prompt({'question': 'What is the capital of France?',
                             'context': 'Document [0](Title: Paris)Paris is the capital '
                                        'of France. It sits on the Seine.'}, 1, args),
        temperature=0, max_tokens=64, timeout=args.request_timeout)
    text = (response.choices[0].text or '').strip()
    if not text:
        raise SystemExit('PREFLIGHT FAILED: the probe request returned no text')
    print('PREFLIGHT OK: %s is available, probe returned %r' % (args.served_model, text[:80]))


def parse_args():
    p = argparse.ArgumentParser(
        description='Context compression with BRIEF-Pro (vLLM) at 0.5x / 1x / 2x.',
        epilog='Start the server first: bash src/compression/serve_brief_pro.sh')
    p.add_argument('--model', dest='served_model', default=MODEL,
                   help='the server\'s --served-model-name, default %(default)s')
    p.add_argument('--tokenizer', default=DEFAULT_TOKENIZER,
                   help='BRIEF-Pro tokenizer (hub id or local directory), used to apply the chat template; '
                        'default %(default)s')
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--base-url', default=DEFAULT_BASE_URL)
    p.add_argument('--api-key', default='EMPTY', help='not checked by vLLM')
    p.add_argument('--prompt-variant', choices=sorted(INSTRUCTION), default='simple',
                   help='simple: the official BRIEF-Pro instruction (paper setting); '
                        'disambiguation: an extended instruction about ambiguous entities')
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--max-concurrency', type=int, default=DEFAULT_MAX_CONCURRENCY)
    p.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument('--request-timeout', type=float, default=DEFAULT_REQUEST_TIMEOUT)
    p.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument('--temperature', type=float, default=0.0,
                   help='default greedy, as in the rest of the benchmark; 0.01 matches the official script')
    p.add_argument('--top-p', type=float, default=None, help='0.95 matches the official script')
    p.add_argument('--setting', choices=['all', 'parallel', 'conditional', 'imbalanced'],
                   default='all')
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
    args.family = 'brief-pro'
    args.enable_thinking = False
    return args


async def main():
    args = parse_args()
    install_seam()
    tasks = [t for t in rc.all_tasks(args.data_root, args.output_root,
                                     args.output_dir, 'without_thinking')
             if rc.selected(t, args)]
    print('Selected %d task(s) for %s (vllm completions, no ICL, no system prompt).'
          % (len(tasks), args.served_model))
    missing = [str(t.input_file) for t in tasks if not t.input_file.exists()]
    if missing:
        raise SystemExit('Missing input files:\n  ' + '\n  '.join(sorted(set(missing))))

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url, max_retries=0)
    try:
        if not args.skip_preflight:
            await probe(client, args)
        if args.dry_run:
            for t in tasks:
                rc.print_task(t, args)
            print('\nExample prompt (%s / k=%d):\n%s'
                  % (tasks[0].name, tasks[0].k,
                     build_prompt(rc.load_jsonl(tasks[0].input_file)[0], tasks[0].k, args)[:1200]))
            return
        for t in tasks:
            # family drives the thinking / channel handling for chat models in
            # run_compression; the completions path does not use it.
            await rc.run_task(t, args, None, client)
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(main())
