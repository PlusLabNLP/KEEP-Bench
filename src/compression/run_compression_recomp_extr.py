#!/usr/bin/env python3
"""Run the compression benchmark through RECOMP extractive (fangyuan/hotpotqa_extractive_compressor).

The model is the sentence selector released with RECOMP: a BertModel (110M parameters,
``max_position_embeddings=512``) initialized from ``facebook/contriever-msmarco`` and
fine-tuned on HotpotQA. It does not generate text. It embeds the query and each sentence,
ranks sentences by dot product, and joins the top k in their original order. Unlike RECOMP
abstractive it has a k parameter, so it runs all three budgets (0.5x/1x/2x) with the
standard k table: 27 output files with 12,306 rows in total.

Configuration
-------------
* Sentence splitting: ``spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'lemmatizer'])``.
  The parser is enabled, so ``doc.sents`` comes from the dependency parse. EXIT uses
  spaCy's ``senter`` instead, so sentence boundaries can differ between the two.
* Scoring unit text: ``f"{title} {sentence}"``, with the paragraph title prepended to every
  sentence. Paragraphs in the parallel and conditional settings have titles; science and
  policy paragraphs (``{idx, label, text}``) do not, and their units are bare sentences.
* Second split: before scoring, each unit is split again with
  ``re.split(r'(?<=[.!?])\\s+')``. This is usually a no-op for a single sentence, but
  titles that contain a period ("St. Louis ...") split the unit, which changes the
  candidate set and the meaning of k. The split stage reports how often this happens.
* Scoring: the query is encoded once; sentences are encoded in batches of 32; embeddings
  are masked mean pooling over ``last_hidden_state``; the score is the dot product
  (no normalization); fp32.
* Selection: global top-k across all paragraphs (not per paragraph), restored to original
  order by index and joined with ``" "``.

Stages
------
``split`` runs spaCy on CPU and writes a sentence cache. ``score`` loads the encoder on one
GPU, reads the cache, scores each row once and writes all three budgets from the same scores.

Usage
-----
    # 1) Sentence splitting (CPU, about 4,102 rows x 22 paragraphs)
    python src/compression/run_compression_recomp_extr.py --stage split --dry-run
    python src/compression/run_compression_recomp_extr.py --stage split

    # 2) Scoring and selection (one GPU; writes all three budgets)
    CUDA_VISIBLE_DEVICES=0 python src/compression/run_compression_recomp_extr.py --stage score --dry-run
    CUDA_VISIBLE_DEVICES=0 python src/compression/run_compression_recomp_extr.py --stage score

The sentence cache is written to ``outputs/_recomp_extractive_sentences/<subset>.jsonl`` and
the compressed outputs to ``outputs/recomp-extractive/contriever/without_thinking/<budget>/<subset>.jsonl``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('run_compression', HERE.parent / 'compression' / 'run_compression.py')
rc = importlib.util.module_from_spec(_spec)
sys.modules['run_compression'] = rc
_spec.loader.exec_module(rc)

MODEL = 'fangyuan/hotpotqa_extractive_compressor'
OUTPUT_DIR = 'recomp-extractive/contriever'
SENTENCE_CACHE_DIR = '_recomp_extractive_sentences'
DEFAULT_SENTENCE_BATCH = 32
# spaCy configuration: parser enabled; ner, tagger and lemmatizer disabled.
SPACY_DISABLE = ['ner', 'tagger', 'lemmatizer']
# Applied to every scoring unit before scoring (see the module docstring).
SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


# --------------------------------------------------------------------------- #
# Subsets: 9 subsets, each with three values of k
# --------------------------------------------------------------------------- #

def subsets(args):
    """Return [(stem, input file, {budget: k})]. All budgets of a subset share one input file."""
    plan = {}
    for task in rc.all_tasks(args.data_root, args.output_root, args.output_dir,
                             'without_thinking'):
        if not rc.selected(task, args):
            continue
        stem = task.output_file.stem
        entry = plan.setdefault(stem, [stem, task.input_file, {}])
        if entry[1] != task.input_file:
            raise SystemExit('%s maps to two input files: %s and %s'
                             % (stem, entry[1], task.input_file))
        if task.budget in entry[2] and entry[2][task.budget] != task.k:
            raise SystemExit('%s budget %s has two different values of k: %d and %d'
                             % (stem, task.budget, entry[2][task.budget], task.k))
        entry[2][task.budget] = task.k
    return [(stem, path, ks) for stem, path, ks in plan.values()]


def cache_path(args, stem: str) -> Path:
    return args.output_root / SENTENCE_CACHE_DIR / ('%s.jsonl' % stem)


def output_path(args, stem: str, budget: str) -> Path:
    return (args.output_root / args.output_dir / 'without_thinking' / budget
            / ('%s.jsonl' % stem))


# --------------------------------------------------------------------------- #
# Stage split: sentence splitting
# --------------------------------------------------------------------------- #

def source_file(formatted: Path) -> Path:
    """The structured ``<name>.paragraphs.jsonl`` next to an input file (provides paragraphs)."""
    return formatted.with_name(formatted.stem + '.paragraphs.jsonl')


def load_paired(formatted: Path):
    """Read (input rows, structured rows) and require matching ids row by row.

    A misalignment would attach sentences to the wrong question.
    """
    rows = rc.load_jsonl(formatted)
    source = source_file(formatted)
    if not source.exists():
        raise SystemExit('Missing structured source file %s (it provides paragraphs)' % source)
    srows = rc.load_jsonl(source)
    if len(rows) != len(srows):
        raise SystemExit('%s has %d rows, %s has %d rows'
                         % (formatted.name, len(rows), source.name, len(srows)))
    for i, (a, b) in enumerate(zip(rows, srows)):
        if a.get('id') != b.get('id'):
            raise SystemExit('id mismatch at row %d: %r vs %r' % (i, a.get('id'), b.get('id')))
        if not b.get('paragraphs'):
            raise SystemExit('%s row %d (%s) has no paragraphs' % (source.name, i, b.get('id')))
    return rows, srows


def row_units(nlp, srow):
    """Return (scoring units, number of extra pieces produced by the second split).

    Units are ordered by paragraph, then by sentence within a paragraph, which is the
    original text order. Selection restores this order by sorting indices.
    """
    units, extra = [], 0
    for p in srow['paragraphs']:
        title = (p.get('title') or '').strip()
        text = p.get('paragraph_text', p.get('text', '')) or ''
        for sentence in nlp(text).sents:
            sentence = sentence.text.strip()
            if not sentence:
                continue
            joined = '%s %s' % (title, sentence) if title else sentence
            pieces = [x.strip() for x in SPLIT_RE.split(joined) if x.strip()]
            extra += len(pieces) - 1
            units.extend(pieces)
    return units, extra


def run_split(args):
    import spacy

    work = subsets(args)
    print('Selected %d subset(s) for sentence splitting.' % len(work))
    missing = [str(p) for _, p, _ in work if not p.exists()]
    if missing:
        raise SystemExit('Missing input files:\n  ' + '\n  '.join(sorted(set(missing))))

    if args.dry_run:
        total = 0
        for stem, in_path, ks in work:
            n = len(rc.load_jsonl(in_path))
            total += n
            cache = cache_path(args, stem)
            print('  %-34s %5d rows  k=%s -> %s%s'
                  % (stem, n, '/'.join(str(ks[b]) for b in rc.BUDGETS),
                     cache.relative_to(args.output_root),
                     ' (exists)' if cache.exists() else ''))
        print('Total %d rows to split; spaCy en_core_web_sm, disable=%s' % (total, SPACY_DISABLE))
        return

    nlp = spacy.load('en_core_web_sm', disable=SPACY_DISABLE)
    for stem, in_path, _ in work:
        cache = cache_path(args, stem)
        if cache.exists() and not args.overwrite:
            print('SKIP existing cache: %s' % cache)
            continue
        rows, srows = load_paired(in_path)
        if args.limit_per_task is not None:
            rows, srows = rows[:args.limit_per_task], srows[:args.limit_per_task]
        out, total, extra = [], 0, 0
        for row, srow in zip(tqdm(rows, desc=stem), srows):
            units, e = row_units(nlp, srow)
            if not units:
                raise SystemExit('%s row %s produced no sentences' % (stem, row.get('id')))
            out.append({'id': row.get('id'), 'query': row.get('question', ''),
                        'units': units})
            total += len(units)
            extra += e
        rc.write_jsonl(cache, out)
        print('DONE: %s  %d rows / %d units (%.1f per row), %d extra pieces from the second split'
              % (cache, len(out), total, total / len(out), extra))


# --------------------------------------------------------------------------- #
# Stage score: Contriever scoring and global top-k
# --------------------------------------------------------------------------- #

def load_model(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit(
            'CUDA is not available (torch %s). The score stage needs a GPU:\n'
            '  CUDA_VISIBLE_DEVICES=0 python %s --stage score'
            % (torch.__version__, Path(sys.argv[0]).name))
    if torch.cuda.device_count() != 1:
        raise SystemExit('Found %d GPUs. Select one with CUDA_VISIBLE_DEVICES.'
                         % torch.cuda.device_count())
    device = torch.device('cuda')
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model, cache_dir=args.cache_dir).to(device).eval()
    print('Loaded %s on %s (%s, fp32)'
          % (args.model, torch.cuda.get_device_name(0), model.config.architectures))
    return tok, model, device


def mean_pooling(token_embeddings, mask):
    """Zero padded positions and divide by the true length; the result does not depend on batch size."""
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def score_units(tok, model, device, query, units, batch_size):
    import torch

    with torch.no_grad():
        enc = tok(query, return_tensors='pt', truncation=True, padding=True).to(device)
        query_embedding = mean_pooling(model(**enc).last_hidden_state,
                                       enc['attention_mask'])[0]
        scores = []
        for i in range(0, len(units), batch_size):
            enc = tok(units[i:i + batch_size], return_tensors='pt',
                      truncation=True, padding=True).to(device)
            embeddings = mean_pooling(model(**enc).last_hidden_state, enc['attention_mask'])
            scores.extend(torch.matmul(embeddings, query_embedding).cpu().tolist())
    return scores


def compose(units, scores, k):
    """Global top-k, restored to original order by index, joined with a single space."""
    import torch

    k = min(k, len(units))
    chosen = sorted(torch.topk(torch.tensor(scores), k=k).indices.tolist())
    return ' '.join(units[i] for i in chosen), len(chosen)


def run_score(args):
    work = subsets(args)
    print('Selected %d subset(s) x %d budgets for %s.'
          % (len(work), len(rc.BUDGETS), args.model))

    plans = []
    for stem, in_path, ks in work:
        cache = cache_path(args, stem)
        outs = {b: output_path(args, stem, b) for b in rc.BUDGETS}
        plans.append((stem, in_path, ks, cache, outs))

    missing = [str(c) for _, _, _, c, _ in plans if not c.exists()]
    if missing:
        raise SystemExit('Missing sentence cache; run --stage split first:\n  '
                         + '\n  '.join(missing))

    if args.dry_run:
        total_rows = total_units = 0
        for stem, in_path, ks, cache, outs in plans:
            rows = rc.load_jsonl(cache)
            units = sum(len(r['units']) for r in rows)
            total_rows += len(rows)
            total_units += units
            done = [b for b in rc.BUDGETS if outs[b].exists()]
            print('  %-34s %5d rows %7d units (%5.1f per row)  k=%s%s'
                  % (stem, len(rows), units, units / len(rows),
                     '/'.join(str(ks[b]) for b in rc.BUDGETS),
                     ('  exists: ' + ','.join(done)) if done else ''))
        print('Total %d rows / %d units (%.1f per row); %d files / %d rows to write'
              % (total_rows, total_units, total_units / total_rows,
                 len(plans) * len(rc.BUDGETS), total_rows * len(rc.BUDGETS)))
        print('Each row is scored once and the scores are shared by all budgets; '
              'sentence batch=%d' % args.sentence_batch_size)
        stem, in_path, ks, cache, outs = plans[0]
        sample = rc.load_jsonl(cache)[0]
        print('\nSample (first row of %s, %d scoring units):\n  query: %s\n  unit[0]: %s\n  unit[1]: %s'
              % (stem, len(sample['units']), sample['query'][:120],
                 sample['units'][0][:160], sample['units'][1][:160]))
        return

    tok, model, device = load_model(args)
    for stem, in_path, ks, cache, outs in plans:
        pending = [b for b in rc.BUDGETS
                   if args.overwrite or not outs[b].exists()
                   or (args.retry_errors and any(rc.is_failed(r)
                                                 for r in rc.load_jsonl(outs[b])))]
        if not pending:
            print('SKIP existing output: %s' % stem)
            continue

        rows = rc.load_jsonl(in_path)
        if args.limit_per_task is not None:
            rows = rows[:args.limit_per_task]
        cached = rc.load_jsonl(cache)
        if len(rows) != len(cached):
            raise SystemExit('%s: input has %d rows but the cache has %d; the cache is stale, '
                             'rerun --stage split with --overwrite'
                             % (stem, len(rows), len(cached)))
        for row, entry in zip(rows, cached):
            if row.get('id') != entry.get('id'):
                raise SystemExit('%s: cache and input ids do not match (%r vs %r)'
                                 % (stem, row.get('id'), entry.get('id')))

        results = {b: [] for b in pending}
        for row, entry in zip(tqdm(rows, desc='%s %s' % (stem, ','.join(pending))), cached):
            units = entry['units']
            scores = score_units(tok, model, device, entry['query'], units,
                                 args.sentence_batch_size)
            for budget in pending:
                text, selected = compose(units, scores, ks[budget])
                out = dict(row)
                out['compressed_context'] = text if text else rc.ERROR_COMPRESSED_CONTEXT
                out['recomp_sentences_total'] = len(units)
                out['recomp_selected'] = selected
                out['recomp_k'] = ks[budget]
                results[budget].append(out)

        for budget in pending:
            rc.write_jsonl(outs[budget], results[budget])
            failed = sum(1 for r in results[budget] if rc.is_failed(r))
            print('DONE: %s%s' % (outs[budget], '  (%d failed)' % failed if failed else ''))


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description='Compression with RECOMP extractive (Contriever sentence selector).')
    p.add_argument('--stage', required=True, choices=['split', 'score'],
                   help='split: sentence splitting with spaCy (CPU); '
                        'score: scoring and selection (GPU)')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--cache-dir', default=None,
                   help='Hugging Face cache directory (default: the standard HF cache)')
    p.add_argument('--output-dir', default=OUTPUT_DIR)
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--sentence-batch-size', type=int, default=DEFAULT_SENTENCE_BATCH,
                   help='sentences per encoder batch (default: %(default)s); mean pooling '
                        'is masked, so the value does not affect results')
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
    args.budget = 'all'          # read by rc.selected; all three budgets are produced
    args.family = 'recomp-extractive'
    args.enable_thinking = False
    return args


def main():
    args = parse_args()
    if args.stage == 'split':
        run_split(args)
    else:
        run_score(args)


if __name__ == '__main__':
    main()
