#!/usr/bin/env python3
"""Run the compression benchmark through RECOMP abstractive (fangyuan/hotpotqa_abstractive).

This compressor has no sentence budget. It is a T5-large model fine-tuned to emit one
summary, and the summary length is whatever ``generate`` produces; there is no parameter
that corresponds to "k sentences". The script therefore runs each of the 9 subsets once
and writes the result under a single budget label (``--budget-label``, default ``fixed``)
instead of the 0.5x/1x/2x budgets. This gives 9 output files with 4,102 rows in total.

The model is an encoder-decoder T5, so it is loaded in-process with transformers rather
than served by vLLM, and rows are run in batches on one GPU. Select the GPU with
``CUDA_VISIBLE_DEVICES``; the script requires exactly one visible device.

Input length
------------
``config.json`` sets ``n_positions: 512``: the model was trained on short retrieved
passages, while the benchmark contexts average about 3,600 tokens and reach about 5,600.
T5 uses relative position embeddings, so longer inputs run, but the model operates far
beyond its training length, and encoder self-attention is quadratic in input length.

``--max-input-tokens`` sets the truncation length:
  * ``8192`` (default) truncates almost no row.
  * ``512`` is the model's training length. Most of each context is cut.
Each output row records ``recomp_input_tokens`` (untruncated prompt length) and
``recomp_truncated``, and the run ends by printing the number of truncated rows.

Generation uses beam search (5 beams), length penalty 2.0, at most 300 new tokens, fp32.

Usage
-----
All commands need a CUDA GPU.

    CUDA_VISIBLE_DEVICES=0 python src/compression/run_compression_recomp_abst.py --dry-run
    CUDA_VISIBLE_DEVICES=0 python src/compression/run_compression_recomp_abst.py
    CUDA_VISIBLE_DEVICES=0 python src/compression/run_compression_recomp_abst.py --retry-errors

Outputs are written to ``outputs/recomp-abstractive/t5-large/without_thinking/fixed/<task>.jsonl``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('run_compression', HERE.parent / 'compression' / 'run_compression.py')
rc = importlib.util.module_from_spec(_spec)
sys.modules['run_compression'] = rc
_spec.loader.exec_module(rc)

MODEL = 'fangyuan/hotpotqa_abstractive'
OUTPUT_DIR = 'recomp-abstractive/t5-large'
# Generation settings: 5 beams, length penalty 2.0, at most 300 new tokens.
DEFAULT_NUM_BEAMS = 5
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_LENGTH_PENALTY = 2.0
DEFAULT_MAX_INPUT_TOKENS = 8192
DEFAULT_BATCH_SIZE = 4
# A batch of 8 that includes 5K-token rows needs about 14 GB and runs out of memory.
# 24576 tokens corresponds to about 8 x 3K or 4 x 6K.
DEFAULT_MAX_BATCH_TOKENS = 24576


def build_prompt(entry) -> str:
    """Prompt format of the RECOMP abstractive compressor."""
    return 'Question: %s\n Document: %s\n Summary: ' % (
        entry.get('question', ''), entry.get('context', ''))


def load_model(args):
    # Fail explicitly instead of falling back to CPU, which would be far too slow.
    if not torch.cuda.is_available():
        raise SystemExit(
            'CUDA is not available (torch %s). This script needs a GPU:\n'
            '  CUDA_VISIBLE_DEVICES=0 python %s'
            % (torch.__version__, Path(sys.argv[0]).name))
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            'Found %d GPUs. Select one with CUDA_VISIBLE_DEVICES, e.g.\n'
            '  CUDA_VISIBLE_DEVICES=0 python %s'
            % (torch.cuda.device_count(), Path(sys.argv[0]).name))
    device = torch.device('cuda')
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    # transformers 5 takes dtype=, transformers 4.x takes torch_dtype=.
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, cache_dir=args.cache_dir, dtype=args.dtype).to(device)
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, cache_dir=args.cache_dir, torch_dtype=args.dtype).to(device)
    model.eval()
    print('Loaded %s on %s (dtype=%s, n_positions=%s)'
          % (args.model, device, args.dtype, getattr(model.config, 'n_positions', '?')))
    return tok, model, device


@torch.no_grad()
def compress_batch(tok, model, device, prompts, args):
    """Return (summaries, input token count per row, whether each row was truncated)."""
    enc = tok(prompts, return_tensors='pt', padding=True,
              truncation=True, max_length=args.max_input_tokens)
    # Truncation is detected from the untruncated length, measured separately.
    full = [len(tok(p, truncation=False)['input_ids']) for p in prompts]
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        enc['input_ids'], attention_mask=enc.get('attention_mask'),
        max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
        length_penalty=args.length_penalty, early_stopping=True)
    texts = tok.batch_decode(out, skip_special_tokens=True)
    return ([t.strip() for t in texts], full,
            [n > args.max_input_tokens for n in full])


def plan_batches(rows, idxs, tok, args):
    """Group rows into batches by a token budget rather than a fixed row count.

    A fixed batch size runs out of memory on this data: T5 encoder self-attention is
    quadratic in length, and a few 5K-token contexts in one batch of 8 with 5 beams need
    about 14 GB. Rows are sorted by token length so that rows of similar length share a
    batch, which also minimizes padding. A batch is closed when its longest row times its
    size would exceed ``--max-batch-tokens`` or its size would exceed ``--batch-size``.
    Returns lists of original row indices; the caller restores row order.
    """
    lengths = {i: len(tok(build_prompt(rows[i]), truncation=False)['input_ids'])
               for i in idxs}
    batches, cur, cur_max = [], [], 0
    for i in sorted(idxs, key=lambda j: lengths[j]):
        nxt_max = max(cur_max, min(lengths[i], args.max_input_tokens))
        if cur and (len(cur) + 1 > args.batch_size
                    or nxt_max * (len(cur) + 1) > args.max_batch_tokens):
            batches.append(cur)
            cur, cur_max = [i], min(lengths[i], args.max_input_tokens)
        else:
            cur.append(i)
            cur_max = nxt_max
    if cur:
        batches.append(cur)
    return batches


def subsets(args):
    """The 9 subsets. Each input file is run once; there is no budget dimension."""
    seen = {}
    for task in rc.all_tasks(args.data_root, args.output_root, args.output_dir,
                             'without_thinking'):
        if not rc.selected(task, args):
            continue
        stem = task.output_file.stem
        if stem in seen:
            continue                      # 0.5x/1x/2x of a subset share the same input
        out = (args.output_root / args.output_dir / 'without_thinking'
               / args.budget_label / ('%s.jsonl' % stem))
        seen[stem] = (stem, task.input_file, out)
    return list(seen.values())


def run_subset(stem, in_path, out_path, tok, model, device, args):
    existing = None
    if out_path.exists():
        if args.retry_errors:
            existing = rc.load_jsonl(out_path)
            if len(existing) != sum(1 for l in in_path.open() if l.strip()):
                raise SystemExit('--retry-errors: row count of %s does not match the input; '
                                 'use --overwrite' % out_path)
            if not any(rc.is_failed(r) for r in existing):
                print('SKIP (no failed rows): %s' % out_path)
                return 0, 0
        elif not args.overwrite:
            print('SKIP existing output: %s' % out_path)
            return 0, 0

    rows = rc.load_jsonl(in_path)
    if args.limit_per_task is not None:
        rows = rows[:args.limit_per_task]

    if existing is None:
        results = [None] * len(rows)
        todo = list(range(len(rows)))
    else:
        results = list(existing[:len(rows)])
        todo = [i for i in range(len(rows)) if rc.is_failed(results[i])]
    print('%s\n  input:  %s\n  output: %s\n  rows=%d%s'
          % (stem, in_path, out_path, len(rows),
             (', retrying %d failed rows' % len(todo)) if existing is not None else ''))

    failed = truncated = 0
    batches = plan_batches(rows, todo, tok, args)
    for idxs in tqdm(batches, desc=stem):
        prompts = [build_prompt(rows[i]) for i in idxs]
        try:
            texts, lens, cuts = compress_batch(tok, model, device, prompts, args)
        except Exception as error:                      # noqa: BLE001 -- recorded per row
            print('Error in %s at rows %s: %s: %s'
                  % (stem, idxs, type(error).__name__, error))
            for i in idxs:
                out = dict(rows[i])
                out['compressed_context'] = rc.ERROR_COMPRESSED_CONTEXT
                results[i] = out
                failed += 1
            continue
        for i, text, n, cut in zip(idxs, texts, lens, cuts):
            out = dict(rows[i])
            out['compressed_context'] = text if text else rc.ERROR_COMPRESSED_CONTEXT
            out['recomp_input_tokens'] = n
            out['recomp_truncated'] = bool(cut)
            results[i] = out
            failed += not text
            truncated += cut

    rc.write_jsonl(out_path, results)
    print('DONE: %s%s' % (out_path, ('  (%d failed)' % failed) if failed else ''))
    return failed, truncated


def parse_args():
    p = argparse.ArgumentParser(
        description='Compression with RECOMP abstractive (T5, no budget tiers).')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--cache-dir', default=None,
                   help='Hugging Face cache directory (default: the standard HF cache)')
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--budget-label', default='fixed',
                   help='path component used in place of 0.5x/1x/2x (default: %(default)s)')
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                   help='maximum rows per batch; batches of long rows are smaller')
    p.add_argument('--max-batch-tokens', type=int, default=DEFAULT_MAX_BATCH_TOKENS,
                   help='token budget per batch (longest row x batch size), '
                        'to avoid running out of memory (default: %(default)s)')
    p.add_argument('--max-input-tokens', type=int, default=DEFAULT_MAX_INPUT_TOKENS,
                   help='input truncation length; 512 is the model\'s training length '
                        '(default: %(default)s)')
    p.add_argument('--max-new-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument('--num-beams', type=int, default=DEFAULT_NUM_BEAMS)
    p.add_argument('--length-penalty', type=float, default=DEFAULT_LENGTH_PENALTY)
    p.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16', 'float16'],
                   help='T5 is prone to NaNs in fp16 (default: float32)')
    p.add_argument('--setting', choices=['all', 'parallel', 'conditional', 'imbalanced'],
                   default='all')
    p.add_argument('--hop', choices=['all', *rc.HOPS], default='all')
    p.add_argument('--domain', choices=['all', 'science', 'policy'], default='all')
    p.add_argument('--perspective', choices=['all', *rc.PERSPECTIVES], default='all')
    p.add_argument('--limit-per-task', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.overwrite and args.retry_errors:
        p.error('--overwrite and --retry-errors are mutually exclusive')
    args.budget = 'all'          # read by rc.selected; no filtering by budget here
    args.family = 'recomp-abstractive'
    args.enable_thinking = False
    return args


def main():
    args = parse_args()
    args.dtype = getattr(torch, args.dtype)
    work = subsets(args)
    print('Selected %d subset(s) for %s (no budget tiers, label=%r).'
          % (len(work), args.model, args.budget_label))
    missing = [str(p) for _, p, _ in work if not p.exists()]
    if missing:
        raise SystemExit('Missing input files:\n  ' + '\n  '.join(sorted(set(missing))))

    if args.dry_run:
        total = 0
        for stem, in_path, out in work:
            n = sum(1 for l in in_path.open() if l.strip())
            total += n
            print('  %-42s %5d rows -> %s%s'
                  % (stem, n, out.relative_to(args.output_root),
                     '' if not out.exists() else ' (exists)'))
        print('Total %d rows to compress; max_input_tokens=%d, beams=%d'
              % (total, args.max_input_tokens, args.num_beams))
        print('\nSample prompt (first 300 characters):\n%s'
              % build_prompt(rc.load_jsonl(work[0][1])[0])[:300])
        return

    tok, model, device = load_model(args)
    tot_failed = tot_cut = 0
    for stem, in_path, out in work:
        f, c = run_subset(stem, in_path, out, tok, model, device, args)
        tot_failed += f
        tot_cut += c
    print('\nAll done: %d failed rows, %d rows truncated (>%d tokens)'
          % (tot_failed, tot_cut, args.max_input_tokens))


if __name__ == '__main__':
    main()
