#!/usr/bin/env python3
"""ICL x thinking ablation at the 1x budget for the four Gemma-4 models.

This is the ablation reported in the paper's appendix. The main run
(``run_compression.py``) is one cell of a 2x2 grid: with ICL, without thinking.
This script runs the other three:

    condition          ICL   thinking   note
    no_icl_no_think     -       -       zero-shot, budget moves into the system prompt
    no_icl_think        -       yes
    icl_think          yes      yes
    icl_no_think       yes      -       same as the main run; not run by default

The scope is 1x only and Gemma-4 only (E2B / E4B / 12B / 31B). 1x is the default
budget in the paper, and running all three budgets would triple the cost.

    9 subsets x 1 budget = 9 tasks per condition (4102 rows);
    3 conditions -> 12306 requests per model

Task construction, the ICL loader, preflight, retries, the ``compressed_context``
output key and the ``ERROR_COMPRESSED_CONTEXT`` sentinel all come from
``run_compression.py``, so the two scripts share one implementation. This module
rebinds only the functions that assemble the prompt and the request body
(``build_messages``, ``request_kwargs`` and ``load_icl_examples``); see
``install_seam`` below.

Dropping the ICL changes the prompt
-----------------------------------
Without demonstrations, a sentence budget in a trailing parenthetical has no
example of what a ``k``-sentence answer looks like. The zero-shot conditions
therefore move the budget into the system prompt and end the user turn bare:

    with ICL     system: <task instructions>                      (no budget)
                 user:   ... ### This is the summary (<must ...>; total length must not
                         exceed {k} sentences):
                 + 2*shots demonstration turns

    without ICL  system: <task instructions>
                         The total length of the summary must not exceed {k} sentences.
                 user:   ... ### This is the summary:

The zero-shot system prompt is the shared system prompt plus that one line, so it
is composed from ``run_compression.system_prompt`` rather than copied.

Gemma-4 thinking
----------------
Two request settings are required:

1. ``chat_template_kwargs={"enable_thinking": True}``. Without it the Gemma
   template pre-fills and immediately closes the thought channel.
2. ``skip_special_tokens=False``. Gemma-4 marks the thought/final boundary with
   special channel tokens (``<|channel>thought`` ... ``<channel|>`` ...
   ``<|channel>final``). If the server strips special tokens, the boundary is lost
   and the hidden reasoning lands in ``compressed_context``.
   ``extract_output_text`` splits on ``<channel|>``, so those tokens must survive.

A server that ignored ``enable_thinking`` would make the thinking arms identical
to the non-thinking ones, so the thinking conditions first run a preflight probe:
an arithmetic question that calls for the thought channel. The probe asserts that
the response stopped normally, showed a thought channel and produced a final
answer.

Usage
-----
    python src/compression/run_ablation.py --list-conditions
    python src/compression/run_ablation.py --model gemma4-e4b          # the 3 extra cells
    python src/compression/run_ablation.py --model gemma4-e4b --condition no_icl_no_think
    python src/compression/run_ablation.py --model gemma4-31b --dry-run

Outputs are written to::

    outputs/ablation/<model dir>/<with|without>_icl/<with|without>_thinking/1x/<task>.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('run_compression', HERE.parent / 'compression' / 'run_compression.py')
rc = importlib.util.module_from_spec(_spec)
sys.modules['run_compression'] = rc
_spec.loader.exec_module(rc)

BUDGET = '1x'
GEMMA_ALIASES = sorted(a for a, s in rc.MODELS.items() if s.family == 'gemma4')

# (with_icl, with_thinking)
CONDITIONS = {
    'no_icl_no_think': (False, False),
    'no_icl_think': (False, True),
    'icl_think': (True, True),
    'icl_no_think': (True, False),        # same as the main run; excluded from the default set
}
DEFAULT_CONDITIONS = ('no_icl_no_think', 'no_icl_think', 'icl_think')

# Thinking needs room for the thought channel on top of the answer; without thinking the
# main run's 1024 is already generous (largest designed compression is 640 tokens).
MAX_TOKENS = {False: 1024, True: 4096}
PREFLIGHT_MAX_TOKENS = 1024

# Per-model thinking budget. Gemma-4 12B's thought often exceeds 4096 tokens: at that
# cap 10.2% of no_icl_think and 12.2% of icl_think rows ended inside an unclosed thought
# channel, which extract_output_text turns into an empty compressed_context. E2B, E4B
# and 31B rarely reach the cap (11 rows in total) and use 4096. For 12B, the thinking
# arms were generated at 4096 and the rows that came back empty were re-requested at
# 8192 with --retry-errors. The longest prompt (22.5k tokens) plus 8192 output tokens
# needs a window of at least 30.7k.
THINKING_MAX_TOKENS = {'gemma4-12b': 8192}

LIMIT_LINE = 'The total length of the summary must not exceed %d sentences.'


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def zeroshot_system_prompt(prompt_family: str, k: int) -> str:
    """Shared system prompt + the sentence budget, which the user turn no longer carries."""
    return rc.system_prompt(prompt_family) + '\n' + LIMIT_LINE % k


def zeroshot_user_prompt(question: str, context: str) -> str:
    return (
        '### This is the question: %s\n'
        '### These are the documents:\n%s\n'
        '### This is the summary:' % (question, context)
    )


def build_messages_with_icl(entry, k, prompt_family, icl_examples):
    return rc.build_messages(entry, k, prompt_family, icl_examples)


def build_messages_zeroshot(entry, k, prompt_family, icl_examples):
    assert not icl_examples, 'zero-shot conditions must not receive demonstrations'
    return [
        {'role': 'system', 'content': zeroshot_system_prompt(prompt_family, k)},
        {'role': 'user', 'content': zeroshot_user_prompt(entry.get('question', ''),
                                                         entry.get('context', ''))},
    ]


# --------------------------------------------------------------------------- #
# The seam into run_compression
# --------------------------------------------------------------------------- #

_RC_BUILD = rc.build_messages
_RC_KWARGS = rc.request_kwargs
_RC_LOAD_ICL = rc.load_icl_examples


def install_seam(with_icl: bool):
    """Point run_compression's prompt/request assembly at this condition.

    ``run_task`` / ``process_one`` / ``preflight_context`` all reach the prompt, the
    request body and the demonstrations through these three module-level names, so
    rebinding them is enough to run a different condition through the whole of that
    machinery. Nothing else is touched.

    ``load_icl_examples`` has to be seamed too: ``run_task`` calls it unconditionally, so a
    zero-shot condition would otherwise still read the ICL file (and hit the assertion in
    ``build_messages_zeroshot``).
    """
    assert all(callable(f) for f in (_RC_BUILD, _RC_KWARGS, _RC_LOAD_ICL)), \
        'run_compression is missing a function this module rebinds'

    rc.build_messages = (_RC_BUILD if with_icl else build_messages_zeroshot)
    rc.load_icl_examples = (_RC_LOAD_ICL if with_icl else (lambda *a, **k: []))

    def request_kwargs(args, family, messages):
        kw = _RC_KWARGS(args, family, messages)
        if args.enable_thinking:
            # Keep Gemma's channel tokens so the thought/final boundary survives.
            kw.setdefault('extra_body', {})['skip_special_tokens'] = False
        return kw

    rc.request_kwargs = request_kwargs


# --------------------------------------------------------------------------- #
# Raw response log (--raw-log)
# --------------------------------------------------------------------------- #

_RC_PROCESS = rc.process_one
_RC_EXTRACT = rc.extract_output_text
_CURRENT_ROW = contextvars.ContextVar('ablation_current_row', default=None)


def install_raw_log(path: Path):
    """Append every response's raw text -- thought included -- to a JSONL side file.

    The output files keep only the extracted answer, so a thought that never closes leaves
    an empty ``compressed_context`` and no trace of what the model actually generated. This
    records it without touching extraction: ``compressed_context`` is exactly what it would
    be without the flag.

    ``run_task`` schedules each ``process_one`` through ``asyncio.as_completed``, i.e. as its
    own Task with its own context, so a ContextVar set there tells ``extract_output_text``
    which row the response belongs to. Calls made outside a row (the thinking preflight)
    see no row and are not logged. A request that fails every retry produces no response
    and so no record; its row keeps the sentinel as before.
    """
    handle = open(path, 'a', encoding='utf-8')

    async def process_one(idx, entry, client, semaphore, task, args, family, icl_examples):
        _CURRENT_ROW.set((task, idx, entry, args))
        return await _RC_PROCESS(idx, entry, client, semaphore, task, args, family, icl_examples)

    def extract_output_text(response, family):
        answer = _RC_EXTRACT(response, family)
        current = _CURRENT_ROW.get()
        if current is not None:
            task, idx, entry, args = current
            choice = response.choices[0]
            usage = getattr(response, 'usage', None)
            raw = rc.message_to_text(getattr(choice.message, 'content', ''))
            handle.write(json.dumps({
                'output_file': str(task.output_file), 'line': idx, 'id': entry.get('id'),
                'max_tokens': args.max_tokens, 'finish_reason': choice.finish_reason,
                'prompt_tokens': getattr(usage, 'prompt_tokens', None),
                'completion_tokens': getattr(usage, 'completion_tokens', None),
                'thought_closed': rc.GEMMA_THINKING_END in raw,
                'raw': raw, 'compressed_context': answer,
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            }, ensure_ascii=False) + '\n')
            handle.flush()
        return answer

    rc.process_one = process_one
    rc.extract_output_text = extract_output_text
    return handle


# --------------------------------------------------------------------------- #
# Thinking preflight
# --------------------------------------------------------------------------- #

PROBE = (
    'A farmer has 17 rows of 19 trees, removes 23 trees, and divides the remainder among '
    '5 fields. Use your private thought channel to verify the arithmetic, then give a '
    'one-line final answer with the quotient and remainder.'
)


async def preflight_thinking(client, args, family):
    """Confirm enable_thinking=True actually produces a visible thought channel.

    A server that quietly ignores the flag would make the "with thinking" arms identical
    to the "without thinking" ones, and nothing downstream would notice.
    """
    messages = [{'role': 'user', 'content': PROBE}]
    probe_args = argparse.Namespace(**{**vars(args), 'max_tokens': PREFLIGHT_MAX_TOKENS})
    response = await client.chat.completions.create(**rc.request_kwargs(probe_args, family, messages))
    choice = response.choices[0]
    raw = rc.message_to_text(getattr(choice.message, 'content', ''))
    final = rc.extract_output_text(response, family)
    visible = (rc.GEMMA_THINKING_END in raw or rc.GEMMA_THINKING_START in raw
               or '</think>' in raw
               or bool(str(getattr(choice.message, 'reasoning_content', '') or '').strip()))
    if choice.finish_reason != 'stop' or not visible or not final:
        raise SystemExit(
            'THINKING PREFLIGHT FAILED: enable_thinking=True did not give a normally '
            'stopped response with both a visible thought channel and a final answer '
            '(finish_reason=%r, visible_thinking=%s, final=%r).\n  raw: %r'
            % (choice.finish_reason, visible, final[:80], raw[:400])
        )
    print('THINKING PREFLIGHT OK: %s shows a thought channel, final=%r'
          % (args.served_model, final[:90]))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def resolve_max_tokens(args, with_thinking: bool) -> int:
    """--max-tokens wins; otherwise the per-condition default, with the 12B override."""
    if args.explicit_max_tokens is not None:
        return args.explicit_max_tokens
    if with_thinking:
        return THINKING_MAX_TOKENS.get(args.model, MAX_TOKENS[True])
    return MAX_TOKENS[False]


def tasks_for(args, with_icl: bool, with_thinking: bool):
    icl_tag = 'with_icl' if with_icl else 'without_icl'
    think_tag = 'with_thinking' if with_thinking else 'without_thinking'
    out_dir = os.path.join('ablation', args.output_dir, icl_tag)
    return [t for t in rc.all_tasks(args.data_root, args.output_root, out_dir, think_tag)
            if t.budget == BUDGET and rc.selected(t, args)]


def parse_args():
    p = argparse.ArgumentParser(
        description='ICL x thinking ablation at the 1x budget, Gemma-4 only.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--list-conditions', action='store_true')
    p.add_argument('--model', choices=GEMMA_ALIASES,
                   help='Gemma-4 alias; fills in served model, family, base URL, output dir.')
    p.add_argument('--served-model'); p.add_argument('--base-url')
    p.add_argument('--api-key', default='EMPTY'); p.add_argument('--output-dir')
    p.add_argument('--condition', action='append', choices=sorted(CONDITIONS),
                   help='repeatable; default = the three cells the main run does not cover')
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--max-concurrency', type=int, default=rc.DEFAULT_MAX_CONCURRENCY)
    p.add_argument('--max-retries', type=int, default=rc.DEFAULT_MAX_RETRIES)
    p.add_argument('--request-timeout', type=float, default=rc.DEFAULT_REQUEST_TIMEOUT)
    p.add_argument('--max-tokens', type=int, default=None,
                   help='default: %d without thinking, %d with (gemma4-12b: %d)'
                        % (MAX_TOKENS[False], MAX_TOKENS[True], THINKING_MAX_TOKENS['gemma4-12b']))
    p.add_argument('--setting', choices=['all', 'parallel', 'conditional', 'imbalanced'], default='all')
    p.add_argument('--hop', choices=['all', *rc.HOPS], default='all')
    p.add_argument('--domain', choices=['all', 'science', 'policy'], default='all')
    p.add_argument('--perspective', choices=['all', *rc.PERSPECTIVES], default='all')
    p.add_argument('--limit-per-task', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--dry-run', action='store_true',
                   help='preflight and list the selected tasks, then exit')
    p.add_argument('--skip-preflight', action='store_true')
    p.add_argument('--raw-log', type=Path, default=None,
                   help='append each response (thought included, token counts, finish_reason) '
                        'to this JSONL file; compressed_context is unaffected')
    args = p.parse_args()

    if args.list_conditions:
        print('%-18s %-6s %-9s %s' % ('condition', 'ICL', 'thinking', 'note'))
        for name, (icl, think) in CONDITIONS.items():
            note = 'same as run_compression.py; not run by default' if name == 'icl_no_think' else ''
            mark = ' ' if name == 'icl_no_think' else '*'
            print('%s%-17s %-6s %-9s %s' % (mark, name, 'yes' if icl else 'no',
                                            'on' if think else 'off', note))
        print('\n* = run by default; budget fixed at 1x; models: %s' % ', '.join(GEMMA_ALIASES))
        raise SystemExit(0)

    spec = rc.MODELS[args.model] if args.model else None
    for attr in ('served_model', 'base_url', 'output_dir'):
        if getattr(args, attr) is None:
            if spec is None:
                p.error('--%s is required when --model is not given' % attr.replace('_', '-'))
            setattr(args, attr, getattr(spec, attr))
    args.family = 'gemma4'
    if spec is not None and spec.family != 'gemma4':
        p.error('this ablation covers Gemma-4 models only')
    if args.overwrite and args.retry_errors:
        p.error('--overwrite and --retry-errors are mutually exclusive')
    args.condition = args.condition or list(DEFAULT_CONDITIONS)
    args.explicit_max_tokens = args.max_tokens        # None: per-condition default
    # run_compression.selected() filters on args.budget.
    args.budget = BUDGET
    return args


async def main():
    args = parse_args()
    family = rc.FAMILIES['gemma4']
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    raw_log = install_raw_log(args.raw_log) if args.raw_log else None
    try:
        max_model_len = None
        if not args.skip_preflight:
            max_model_len = await rc.preflight_model(client, args)

        for name in args.condition:
            with_icl, with_thinking = CONDITIONS[name]
            args.enable_thinking = with_thinking
            args.max_tokens = resolve_max_tokens(args, with_thinking)
            install_seam(with_icl)

            tasks = tasks_for(args, with_icl, with_thinking)
            print()
            print('=' * 78)
            print('== %s   ICL=%s thinking=%s   %d task(s) (%s)  max_tokens=%d'
                  % (name, 'yes' if with_icl else 'no', 'on' if with_thinking else 'off',
                     len(tasks), BUDGET, args.max_tokens))
            print('=' * 78)
            missing = [str(t.input_file) for t in tasks if not t.input_file.exists()]
            if missing:
                raise SystemExit('Missing input file(s):\n  ' + '\n  '.join(sorted(set(missing))))

            if not args.skip_preflight:
                if with_thinking:
                    await preflight_thinking(client, args, family)
                if max_model_len:
                    await rc.preflight_context(args, tasks, max_model_len)
            if args.dry_run:
                for t in tasks:
                    rc.print_task(t, args)
                continue
            for t in tasks:
                await rc.run_task(t, args, family, client)
    finally:
        await client.close()
        if raw_log is not None:
            raw_log.close()


if __name__ == '__main__':
    asyncio.run(main())
