#!/usr/bin/env python3
"""Run every arm's FLOPs calculator over the whole benchmark and print one table.

The calculators in this directory price one request or group. This driver feeds them the
benchmark as executed: 4102 cases per arm, the prompts the models received, and the token
counts the models produced. It runs in two stages, because the first is slow and cacheable:

    --stage inputs   tokenize and write outputs/_flops_inputs/<arm>.jsonl   (CPU, minutes)
    --stage table    read the cache, price it, print the table              (seconds)
    (no --stage)     inputs if missing, then table

What "as executed" means per arm
--------------------------------
* 11 ICL compressors -- vLLM with continuous batching, so no padding. The ICL block is
  identical across every case of a setting and prefix caching is enabled
  (`src/compression/serve_vllm.sh`), so each setting group pays one prefix prefill and
  every case pays only its own tail. `G` is corrected back to the pre-trim generated
  length (see below).
* EXIT -- one prefix group per document, suffixes sorted by length and padded per batch.
* RECOMP extractive -- 32 candidates per batch in document order, padded to the batch max.
* RECOMP abstractive -- encoder at its full length (max_input_tokens=8192, nothing
  truncated), batches replayed from plan_batches, beams=5.
* BRIEF-Pro -- zero-shot, so there is no reusable prefix and nothing to amortize.

EXIT and RECOMP-extractive score each sentence once and reuse the scores for all three
budgets, so their numbers are budget-independent; the other arms' are not.

Recovering G
------------
`compressed_context` is not exactly what the model generated: looping outputs were trimmed
and rows that hit `max_tokens` were cut back to a sentence boundary. The manifests in
`outputs/_truncation_manifests/` record the pre-trim length, and G is restored from them.
There are three manifest shapes:

    tokens_before/tokens_after  exact pre-trim token count
    tokens (=1024)              the row hit the cap, so G is exactly the cap
    old_chars/new_chars         manual edits with character counts only; G is extrapolated
                                by the row's own chars-per-token and clamped at --max-tokens

The manifest `line` field is 0-based.

Usage
-----
    python src/computation/run_flops.py --stage inputs
    python src/computation/run_flops.py --stage table
    python src/computation/run_flops.py --budget 2x --only qwen3-8b exit-doc
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
REPO = SRC.parent
sys.path.insert(0, str(HERE))


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = load(SRC / 'compression' / 'run_compression.py', 'run_compression')
llm = load(HERE / 'llm_flops.py', 'llm_flops')

CACHE = rc.DEFAULT_OUTPUT_ROOT / '_flops_inputs'
SUBSETS = ('parallel_setting_two_hop', 'parallel_setting_three_hop', 'parallel_setting_four_hop',
           'conditional_setting_two_hop', 'conditional_setting_three_hop',
           'conditional_setting_four_hop', 'science_domain_three_perspectives',
           'science_domain_four_perspectives', 'policy_domain')

# One tokenization per distinct (tokenizer.json, chat_template) pair. Llama-3.1-8B's chat
# template differs from Llama-3.2's, and Gemma-4's E-series template differs from 12B/31B's.
VARIANTS = {
    'qwen': dict(path='Qwen/Qwen3-32B', kwargs={'enable_thinking': False}),
    'llama32': dict(path='meta-llama/Llama-3.2-3B-Instruct', kwargs={}),
    'llama31': dict(path='meta-llama/Llama-3.1-8B-Instruct', kwargs={}),
    'gemmaE': dict(path='google/gemma-4-E2B-it', kwargs={'enable_thinking': False}),
    'gemmaBig': dict(path='google/gemma-4-12B-it', kwargs={'enable_thinking': False}),
}
RECOMP_EXTRACTIVE_TOKENIZER = 'fangyuan/hotpotqa_extractive_compressor'
RECOMP_ABSTRACTIVE_TOKENIZER = 'fangyuan/hotpotqa_abstractive'

# arm -> (outputs/ subdir, llm_flops alias, tokenizer variant)
ICL_ARMS = {
    'qwen3-4b': ('qwen3/4B', 'qwen3-4b', 'qwen'),
    'qwen3-8b': ('qwen3/8B', 'qwen3-8b', 'qwen'),
    'qwen3-14b': ('qwen3/14B', 'qwen3-14b', 'qwen'),
    'qwen3-32b': ('qwen3/32B', 'qwen3-32b', 'qwen'),
    'gemma4-e2b': ('gemma4/E2B', 'gemma4-e2b', 'gemmaE'),
    'gemma4-e4b': ('gemma4/E4B', 'gemma4-e4b', 'gemmaE'),
    'gemma4-12b': ('gemma4/12B', 'gemma4-12b', 'gemmaBig'),
    'gemma4-31b': ('gemma4/31B', 'gemma4-31b', 'gemmaBig'),
    'llama3.2-1b': ('llama3/1B', 'llama3.2-1b', 'llama32'),
    'llama3.2-3b': ('llama3/3B', 'llama3.2-3b', 'llama32'),
    'llama3.1-8b': ('llama3/8B', 'llama3.1-8b', 'llama31'),
}
BASELINE_ARMS = ('exit-doc', 'exit-fullctx', 'brief-pro', 'recomp-extr', 'recomp-abst')
# prompt_family -> the prefix group it shares an ICL block with
GROUP_OF = {'parallel': 'parallel', 'conditional': 'conditional',
            'science': 'science', 'policy': 'policy'}


def tokenizer_at(path: str):
    """Load a tokenizer from a Hugging Face repository id or a local directory."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True)
    except AttributeError:
        # Some configs saved by transformers 5 store extra_special_tokens as a list, which
        # older transformers versions cannot read.
        return AutoTokenizer.from_pretrained(path, use_fast=True,
                                             extra_special_tokens={'video_token': '<|video|>'})


def common_prefix(acc: list[int] | None, ids: list[int]) -> list[int]:
    """Running longest common prefix -- exact over every row, without keeping them all."""
    if acc is None:
        return list(ids)
    n = 0
    for a, b in zip(acc, ids):
        if a != b:
            break
        n += 1
    return acc[:n]


# --------------------------------------------------------------------------- #
# G: undo the post-hoc trimming
# --------------------------------------------------------------------------- #

def truncation_fixes() -> dict[tuple[str, int], tuple[str, float]]:
    fixes: dict[tuple[str, int], tuple[str, float]] = {}
    manifests = rc.DEFAULT_OUTPUT_ROOT / '_truncation_manifests'
    for path in sorted(manifests.glob('*.jsonl')):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['file'], row['line'])          # line is 0-based
            if 'tokens_before' in row:
                fixes[key] = ('abs', row['tokens_before'])
            elif 'tokens' in row:
                fixes[key] = ('abs', row['tokens'])
            elif row.get('new_chars'):
                fixes[key] = ('ratio', row['old_chars'] / row['new_chars'])
    return fixes


def generated_tokens(tok, path: Path, fixes, cap: int):
    rows = [json.loads(l) for l in path.open() if l.strip()]
    texts = [(r.get('compressed_context') or '') for r in rows]
    lens = [max(1, len(x)) for x in tok(texts, add_special_tokens=False).input_ids]
    fixed = 0
    for i in range(len(rows)):
        hit = fixes.get((str(path), i))
        if not hit:
            continue
        want = int(round(lens[i] * hit[1])) if hit[0] == 'ratio' else int(hit[1])
        lens[i] = max(lens[i], min(cap, want))
        fixed += 1
    return lens, fixed


# --------------------------------------------------------------------------- #
# Stage: inputs
# --------------------------------------------------------------------------- #

def build_icl_inputs(args, fixes):
    tasks = [t for t in rc.all_tasks(args.data_root, args.output_root, 'x', 'without_thinking')
             if t.budget == args.budget]
    rows_of = {t.name: [json.loads(l) for l in t.input_file.open() if l.strip()] for t in tasks}

    # One pass per tokenizer variant: prompt lengths and the exact per-group prefix.
    prompts: dict[str, dict] = {}
    for vname, variant in VARIANTS.items():
        tok = tokenizer_at(variant['path'])
        lengths: dict[str, list[int]] = collections.defaultdict(list)
        prefix: dict[str, list[int] | None] = collections.defaultdict(lambda: None)
        for task in tasks:
            icl = rc.load_icl_examples(task, args.data_root, args.shuffle_seed)
            group = GROUP_OF[task.icl_family]
            texts = [tok.apply_chat_template(
                        rc.build_messages(row, task.k, task.prompt_family, icl),
                        tokenize=False, add_generation_prompt=True, **variant['kwargs'])
                     for row in rows_of[task.name]]
            for ids in tok(texts, add_special_tokens=False).input_ids:
                lengths[group].append(len(ids))
                prefix[group] = common_prefix(prefix[group], ids)
        prompts[vname] = {g: {'P': lengths[g], 'prefix': len(prefix[g])} for g in lengths}
        print('  %-9s %s' % (vname, {g: (len(d['P']), d['prefix']) for g, d in prompts[vname].items()}),
              flush=True)

    order = collections.defaultdict(list)
    for task in tasks:
        order[GROUP_OF[task.icl_family]].append(task)

    for arm, (outdir, _alias, vname) in ICL_ARMS.items():
        if args.only and arm not in args.only:
            continue
        tok = tokenizer_at(VARIANTS[vname]['path'])
        out, nfix = [], 0
        for group, group_tasks in order.items():
            P = prompts[vname][group]['P']
            pre = prompts[vname][group]['prefix']
            G: list[int] = []
            for task in group_tasks:
                path = args.output_root / outdir / 'without_thinking' / args.budget / task.output_file.name
                g, k = generated_tokens(tok, path, fixes, args.max_tokens)
                G.extend(g)
                nfix += k
            if len(G) != len(P):
                raise SystemExit('%s %s: %d output rows vs %d prompt rows'
                             % (arm, group, len(G), len(P)))
            for p, g in zip(P, G):
                out.append({'group': group, 'prefix_tokens': pre,
                            'prompt_tokens': p, 'generated_tokens': g})
        write(arm, out)
        print('  %-12s %5d rows, G corrected in %d rows' % (arm, len(out), nfix), flush=True)


def build_exit_inputs(args, scope: str):
    ex = load(SRC / 'compression' / 'run_compression_exit.py', 'run_compression_exit')
    base = json.loads((Path(ex.DEFAULT_MODEL_DIR) / 'READY.json').read_text())['base_path']
    tok = tokenizer_at(base)
    out = []
    for name in SUBSETS:
        rows = [json.loads(l) for l in
                (args.output_root / ex.SENTENCE_CACHE_DIR / (name + '.jsonl')).open() if l.strip()]
        tails = [len(x) for x in tok([ex.PROMPT_TAIL % s for r in rows for s in r['sentences']],
                                     add_special_tokens=False).input_ids]
        if scope == 'document':
            heads = [len(x) for x in tok([ex.PROMPT_HEAD % (r['query'], d)
                                          for r in rows for d in r['docs']]).input_ids]
        else:
            heads = [len(x) for x in tok([ex.PROMPT_HEAD % (r['query'], ' '.join(r['sentences']))
                                          for r in rows]).input_ids]
        ti = hi = 0
        for r in rows:
            n = len(r['sentences'])
            tl = tails[ti:ti + n]; ti += n
            if scope == 'document':
                hl = heads[hi:hi + len(r['docs'])]; hi += len(r['docs'])
                groups = []
                for di in range(len(r['docs'])):
                    idx = [j for j, o in enumerate(r['owner']) if o == di]
                    if idx:
                        groups.append({'prefix_tokens': hl[di],
                                       'suffix_tokens': [tl[j] for j in idx]})
            else:
                groups = [{'prefix_tokens': heads[hi], 'suffix_tokens': tl}]; hi += 1
            out.append({'groups': groups, 'batch_size': ex.DEFAULT_BATCH_SIZE,
                        'max_batch_tokens': ex.DEFAULT_MAX_BATCH_TOKENS,
                        'lora_unmerged': False,          # --stage merge merged the adapter
                        'logit_mode': 'all', 'padding_mode': 'batch-max'})
        if ti != len(tails):
            raise SystemExit('exit %s: sentence count mismatch' % name)
    write('exit-doc' if scope == 'document' else 'exit-fullctx', out)
    print('  %-12s %5d rows, %d prefix groups'
          % ('exit-' + ('doc' if scope == 'document' else 'fullctx'),
             len(out), sum(len(r['groups']) for r in out)), flush=True)


def build_recomp_extr_inputs(args):
    extr = load(SRC / 'compression' / 'run_compression_recomp_extr.py', 'rc_extr')
    tok = tokenizer_at(RECOMP_EXTRACTIVE_TOKENIZER)
    out = []
    for name in SUBSETS:
        for line in (args.output_root / extr.SENTENCE_CACHE_DIR / (name + '.jsonl')).open():
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({'query_tokens': len(tok(r['query'])['input_ids']),
                        'candidate_tokens': [len(x) for x in tok(r['units'])['input_ids']],
                        'padding_mode': 'batch-max',
                        'batch_size': extr.DEFAULT_SENTENCE_BATCH})
    write('recomp-extr', out)
    print('  %-12s %5d rows, %d sentence units'
          % ('recomp-extr', len(out), sum(len(r['candidate_tokens']) for r in out)), flush=True)


def build_brief_inputs(args):
    import types
    bp = load(SRC / 'compression' / 'run_compression_brief.py', 'run_compression_brief')
    bargs = types.SimpleNamespace(prompt_variant='simple', tokenizer=bp.DEFAULT_TOKENIZER)
    tok = bp.tokenizer(bargs)
    out = []
    for task in rc.all_tasks(args.data_root, args.output_root, 'brief-pro/3b', 'without_thinking'):
        if task.budget != args.budget:
            continue
        rows = [json.loads(l) for l in task.input_file.open() if l.strip()]
        orows = [json.loads(l) for l in task.output_file.open() if l.strip()]
        P = [len(x) for x in tok([bp.build_prompt(r, task.k, bargs) for r in rows],
                                add_special_tokens=False).input_ids]
        G = [max(1, len(x)) for x in tok([(o.get('compressed_context') or '') for o in orows],
                                        add_special_tokens=False).input_ids]
        out.extend({'prompt_tokens': p, 'generated_tokens': g} for p, g in zip(P, G))
    write('brief-pro', out)
    print('  %-12s %5d rows' % ('brief-pro', len(out)), flush=True)


def build_recomp_abst_inputs(args):
    ab = load(SRC / 'compression' / 'run_compression_recomp_abst.py', 'rc_abst')
    tok = tokenizer_at(RECOMP_ABSTRACTIVE_TOKENIZER)
    out = []
    for name in SUBSETS:
        rows = [json.loads(l) for l in
                (args.output_root / 'recomp-abstractive/t5-large/without_thinking/fixed'
                 / (name + '.jsonl')).open() if l.strip()]
        # recomp_input_tokens is the untruncated encoder length, recorded by the runner.
        L = [r['recomp_input_tokens'] for r in rows]
        G = [max(1, len(x)) for x in tok([(r.get('compressed_context') or '') for r in rows],
                                        add_special_tokens=False).input_ids]
        # Replay plan_batches: sort by length, then cap on rows and on width*rows.
        batches, cur, cur_max = [], [], 0
        for i in sorted(range(len(rows)), key=lambda j: L[j]):
            nxt = max(cur_max, min(L[i], ab.DEFAULT_MAX_INPUT_TOKENS))
            if cur and (len(cur) + 1 > ab.DEFAULT_BATCH_SIZE
                        or nxt * (len(cur) + 1) > ab.DEFAULT_MAX_BATCH_TOKENS):
                batches.append(cur); cur, cur_max = [i], min(L[i], ab.DEFAULT_MAX_INPUT_TOKENS)
            else:
                cur.append(i); cur_max = nxt
        if cur:
            batches.append(cur)
        for b in batches:
            width = max(min(L[i], ab.DEFAULT_MAX_INPUT_TOKENS) for i in b)
            # One generate() call decodes the whole batch until every row has finished.
            steps = max(G[i] for i in b)
            for _ in b:
                out.append({'encoder_tokens': width, 'decoder_steps': steps,
                            'beam_width': ab.DEFAULT_NUM_BEAMS})
    write('recomp-abst', out)
    print('  %-12s %5d rows' % ('recomp-abst', len(out)), flush=True)


def write(arm: str, rows: list[dict]):
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / (arm + '.jsonl')).open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def run_inputs(args):
    fixes = truncation_fixes()
    print('_truncation_manifests: %d records (line is 0-based)' % len(fixes))
    want = args.only or (list(ICL_ARMS) + list(BASELINE_ARMS))
    if any(a in ICL_ARMS for a in want):
        build_icl_inputs(args, fixes)
    if 'exit-doc' in want:
        build_exit_inputs(args, 'document')
    if 'exit-fullctx' in want:
        build_exit_inputs(args, 'full')
    if 'recomp-extr' in want:
        build_recomp_extr_inputs(args)
    if 'brief-pro' in want:
        build_brief_inputs(args)
    if 'recomp-abst' in want:
        build_recomp_abst_inputs(args)


# --------------------------------------------------------------------------- #
# Stage: table
# --------------------------------------------------------------------------- #

def price_icl(arm: str, alias: str, rows: list[dict]) -> float:
    """Per setting group: every case warm-cache, plus exactly one prefix warm-up."""
    calc = llm.calculator_for(alias)
    total = 0
    for group, members in group_by(rows, 'group').items():
        prefix = members[0]['prefix_tokens']
        for row in members:
            total += llm.prefix_cached_calculate(
                {'prompt_tokens': row['prompt_tokens'],
                 'generated_tokens': row['generated_tokens']},
                prefix_tokens=prefix, base_calculator=calc)['flops']
        total += llm.prefix_cached_calculate(
            {'prompt_tokens': prefix + 1, 'generated_tokens': 1},
            prefix_tokens=prefix, base_calculator=calc)['cached_prefix_prefill_flops']
    return total / len(rows)


def group_by(rows, key):
    out = collections.defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


# arm -> (calculator module, display name, outputs/ subdir)
BASELINE_CALC = {
    'exit-doc': ('exit_flops.py', 'EXIT doc (gemma-2b)', 'exit/gemma-2b-doc'),
    'exit-fullctx': ('exit_flops.py', 'EXIT fullctx (gemma-2b)', 'exit/gemma-2b-fullctx'),
    'brief-pro': ('brief_pro_flops.py', 'BRIEF-pro (3B)', 'brief-pro/3b'),
    'recomp-extr': ('recomp_extractive_flops.py', 'RECOMP extractive (contriever)',
                    'recomp-extractive/contriever'),
    'recomp-abst': ('recomp_abstractive_flops.py', 'RECOMP abstractive (t5-large)',
                    'recomp-abstractive/t5-large'),
}
DROP = {'group', 'prefix_tokens'}


def run_table(args):
    results = []
    for arm, (outdir, alias, _v) in ICL_ARMS.items():
        if args.only and arm not in args.only:
            continue
        rows = read(arm)
        if rows is None:
            continue
        results.append((llm.MODELS[alias]['method'], price_icl(arm, alias, rows) / 1e12,
                        'ICL compressor / vLLM', len(rows), outdir, True))
    for arm in BASELINE_ARMS:
        if args.only and arm not in args.only:
            continue
        rows = read(arm)
        if rows is None:
            continue
        module, label, outdir = BASELINE_CALC[arm]
        calc = load(HERE / module, 'calc_' + arm.replace('-', '_')).calculate
        total = sum(calc({k: v for k, v in row.items() if k not in DROP})['flops']
                    for row in rows)
        results.append((label, total / len(rows) / 1e12, 'standalone calculator', len(rows),
                        outdir, False))

    results.sort(key=lambda r: -r[1])
    print('\n%s / without_thinking  as-executed average TFLOPs per case\n' % args.budget)
    print('  %-32s %10s  %7s  %s' % ('arm', 'TFLOPs/case', 'cases', 'kind'))
    print('  ' + '-' * 72)
    for name, value, kind, n, _outdir, _amortized in results:
        print('  %-32s %10.2f  %7d  %s' % (name, value, n, kind))
    if len(results) > 1:
        print('\n  spread %.0fx (%.2f -> %.2f)'
              % (results[0][1] / results[-1][1], results[-1][1], results[0][1]))
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{'method': d, 'name': n, 'tflops_per_case': v, 'prefix_amortized': a, 'cases': c}
             for n, v, _k, c, d, a in results],
            ensure_ascii=False, indent=2))
        print('\n  wrote %s' % args.json)


def read(arm: str):
    path = CACHE / (arm + '.jsonl')
    if not path.is_file():
        print('  skipping %s (no cached inputs; run --stage inputs first)' % arm)
        return None
    return [json.loads(l) for l in path.open() if l.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--stage', choices=['inputs', 'table'],
                   help='default: both stages (inputs is skipped if its cache exists)')
    p.add_argument('--budget', default='1x', choices=rc.BUDGETS)
    p.add_argument('--only', nargs='+', metavar='ARM',
                   help='restrict to these arms; names as in ICL_ARMS / BASELINE_ARMS')
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--max-tokens', type=int, default=1024,
                   help='--max-tokens used for compression, the upper bound on G '
                        '(default %(default)s)')
    p.add_argument('--json', metavar='PATH', help='also write the table as JSON')
    p.add_argument('--overwrite', action='store_true', help='rebuild existing input caches')
    args = p.parse_args()

    if args.stage in (None, 'inputs'):
        need = args.only or (list(ICL_ARMS) + list(BASELINE_ARMS))
        missing = [a for a in need if not (CACHE / (a + '.jsonl')).is_file()]
        if args.overwrite or missing or args.stage == 'inputs':
            print('=== stage inputs ===')
            run_inputs(args)
        else:
            print('input caches complete; skipping inputs (--overwrite rebuilds them)')
    if args.stage in (None, 'table'):
        run_table(args)


if __name__ == '__main__':
    main()
