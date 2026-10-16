#!/usr/bin/env python3
"""Evidence-position experiment (Section 5.3): parallel two-hop compression under 14 document orderings.

What this runs
--------------
``run_compression.py`` compresses each case once, with the documents in the released order.
This script compresses the same 400 parallel two-hop cases 14 times, with the 4 evidence
documents moved to 14 prescribed layouts while the 18 distractors keep their released relative
order. The inputs are written to ``data/parallel/position/`` by
``src/data/build_position_variants.py``; that script's docstring has the layout table and
describes how the evidence documents are identified.

The budget is fixed at 1x and thinking is off. ``k`` comes from the shared table
(``PARALLEL_K['two_hop']['1x']`` = 4 sentences), so each position run is directly comparable
with the same model's ``parallel_setting_two_hop.jsonl`` at 1x from the main run. The paper
reports three models (``llama3.1-8b``, ``gemma4-e4b``, ``qwen3-8b``); any registry alias works.

    14 settings x 400 cases = 5600 requests per model; 3 models = 16800.

Prompts
-------
The system prompt, the user turn, the sentence budget and the ICL demonstrations are the ones
``run_compression.py`` uses for parallel / two_hop / 1x; only the input files and output stems
differ. This script builds its own ``TaskSpec`` list and calls ``rc.run_task``, so preflight,
retries, resume, ``--retry-errors``, the ``compressed_context`` key and the
``ERROR_COMPRESSED_CONTEXT`` sentinel are shared. Unlike ``run_ablation.py``, it rebinds
nothing.

The position tasks are kept out of ``all_tasks()`` because other scripts enumerate that list
filtered only by budget, and the position experiment is a separate axis.

Output
------
``outputs/<model_dir>/without_thinking/1x/position_<setting>.jsonl``: the main run's 1x
directory, with a ``position_`` stem so the files do not collide. Each row is its input row
plus ``compressed_context``, so it also carries ``position_setting`` and ``evidence_slots``,
which tie a judged row to its layout without the manifest.

Usage
-----
    python src/compression/run_compression_position.py --list-settings
    python src/compression/run_compression_position.py --model qwen3-8b --dry-run
    python src/compression/run_compression_position.py --model qwen3-8b
    python src/compression/run_compression_position.py --model qwen3-8b --only distant_true1
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    """Load a sibling script as a module, reusing one already loaded under this name.

    Re-executing run_compression when another script has already loaded it would create a
    second ``TaskSpec`` class whose instances are not interchangeable with the first.
    ``@dataclass`` also requires the module to be registered in ``sys.modules``.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = _load('run_compression', HERE / 'run_compression.py')
bpv = _load('build_position_variants', HERE.parent / 'data' / 'build_position_variants.py')

BUDGET = '1x'
HOP = 'two_hop'
SETTINGS = bpv.SETTINGS                 # the builder owns the 14 names
INPUT_DIR = Path(bpv.HOP_DIR) / 'position'
STEM = 'position_%s'
# The three models the paper reports. --model accepts any registry alias.
PAPER_ARMS = ('llama3.1-8b', 'gemma4-e4b', 'qwen3-8b')


def position_tasks(args) -> list:
    """One TaskSpec per layout. Everything but input_file / output_file / name is the
    parallel / two_hop / 1x task the main run already builds."""
    k = rc.PARALLEL_K[HOP][BUDGET]
    out_base = args.output_root / args.output_dir / 'without_thinking' / BUDGET
    tasks = []
    for name in SETTINGS:
        tasks.append(rc.TaskSpec(
            name=f'position/{name}/{BUDGET}/{k}sentences',
            setting='parallel',
            input_file=args.data_root / INPUT_DIR / f'{name}.jsonl',
            output_file=out_base / (STEM % name + '.jsonl'),
            k=k,
            prompt_family='parallel',
            icl_family='parallel',
            budget=BUDGET,
            hop=HOP,
        ))
    return tasks


def list_settings(args) -> None:
    manifest = args.data_root / INPUT_DIR / '_manifest.json'
    slots = {}
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        slots = data.get('slots', {})
        print('input: %s (labels %s)' % (manifest.parent,
                                         'renumbered 0..n-1' if data.get('renumbered_labels') else 'original numbering'))
    print('%-24s %-10s %s' % ('setting', 'rows', 'evidence slots (22 documents; first / second block)'))
    for name in SETTINGS:
        path = args.data_root / INPUT_DIR / f'{name}.jsonl'
        rows = sum(1 for line in path.open(encoding='utf-8') if line.strip()) if path.exists() else 0
        layout = name.rsplit('_true', 1)[0]
        block = slots.get(layout, {}).get('22')
        shown = '%s / %s' % tuple(','.join(map(str, b)) for b in block) if block else '-'
        if block and name.endswith('true2'):
            shown = '%s / %s' % tuple(','.join(map(str, b)) for b in (block[1], block[0]))
        print('%-24s %-10s %s' % (name, rows or 'MISSING', shown))
    print('\nbudget fixed at %s, k=%d, thinking off; models reported in the paper: %s'
          % (BUDGET, rc.PARALLEL_K[HOP][BUDGET], ', '.join(PAPER_ARMS)))


def parse_args():
    p = argparse.ArgumentParser(
        description='Evidence-position compression: 14 document layouts of parallel two-hop, 1x, thinking off.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--list-settings', action='store_true', help='List the 14 layouts and their evidence slots, then exit.')
    p.add_argument('--model', choices=sorted(rc.MODELS),
                   help='Registry alias; fills in served model, family, base URL and output dir.')
    p.add_argument('--served-model'); p.add_argument('--family', choices=sorted(rc.FAMILIES))
    p.add_argument('--base-url'); p.add_argument('--output-dir')
    p.add_argument('--api-key', default='EMPTY')
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--only', action='append', choices=list(SETTINGS),
                   help='Repeatable; default: all 14 layouts.')
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--max-concurrency', type=int, default=rc.DEFAULT_MAX_CONCURRENCY)
    p.add_argument('--max-retries', type=int, default=rc.DEFAULT_MAX_RETRIES)
    p.add_argument('--request-timeout', type=float, default=rc.DEFAULT_REQUEST_TIMEOUT)
    p.add_argument('--max-tokens', type=int, default=rc.DEFAULT_MAX_TOKENS,
                   help='Default %d, as in the main run (k=4 outputs are about 150 tokens).'
                        % rc.DEFAULT_MAX_TOKENS)
    p.add_argument('--limit-per-task', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Preflight and list the selected tasks, then exit.')
    p.add_argument('--skip-preflight', action='store_true')
    args = p.parse_args()

    if args.list_settings:
        list_settings(args)
        raise SystemExit(0)

    spec = rc.MODELS[args.model] if args.model else None
    for attr in ('served_model', 'family', 'base_url', 'output_dir'):
        if getattr(args, attr) is None:
            if spec is None:
                p.error('--%s is required when --model is not given' % attr.replace('_', '-'))
            setattr(args, attr, getattr(spec, attr))
    if args.overwrite and args.retry_errors:
        p.error('--overwrite and --retry-errors are mutually exclusive')

    # Thinking is always off here; run_task / request_kwargs read this attribute.
    args.enable_thinking = False
    args.budget = BUDGET
    return args


async def main():
    args = parse_args()
    family = rc.FAMILIES[args.family]
    tasks = [t for t in position_tasks(args)
             if args.only is None or t.name.split('/')[1] in args.only]

    missing = [str(t.input_file) for t in tasks if not t.input_file.exists()]
    if missing:
        raise SystemExit('Missing input file(s) (run src/data/build_position_variants.py first):\n  '
                         + '\n  '.join(sorted(set(missing))))

    rows = sum(sum(1 for line in t.input_file.open(encoding='utf-8') if line.strip()) for t in tasks)
    print('Position experiment: %d task(s) / %d rows  model=%s  budget=%s  k=%d  thinking=off'
          % (len(tasks), rows, args.served_model, BUDGET, rc.PARALLEL_K[HOP][BUDGET]))

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    try:
        if not args.skip_preflight:
            max_model_len = await rc.preflight_model(client, args)
            if max_model_len:
                await rc.preflight_context(args, tasks, max_model_len)
        if args.dry_run:
            for task in tasks:
                rc.print_task(task, args)
            return
        for task in tasks:
            await rc.run_task(task, args, family, client)
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(main())
