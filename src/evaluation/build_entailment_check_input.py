#!/usr/bin/env python3
"""Build entailment-check inputs from the compression outputs.

Every compression output row is expanded into one row per item the entailment judge checks:
a proposition (parallel, conditional) or a perspective (imbalanced: science and policy). Each
judge puts the whole row into its prompt as JSON, so a row holds exactly the fields below and
nothing else:

    parallel      {"context", "proposition"}
                  logical_chain_1 then logical_chain_2, 2 x hop rows per question
    conditional   <stem>_proposition_only.jsonl  {"context", "proposition"}
                  <stem>_key_proposition.jsonl   {"context", "key", "disambiguation_proposition",
                                                  "proposition"}
                  hop rows per question; the key proposition is the one chain element that
                  contains the question's key (exactly one per question, asserted)
    imbalanced    {"question", "compressed_context", "perspective"}, one row per perspective

Questions, chains, keys and perspectives are read from the data files under ``data/``
(``--data-root``). Every compression output row is checked against its data row (same id,
same fields, same order), so a misaligned output file fails instead of pairing contexts with
the wrong propositions.

The input rows carry no id, so two side files record where each row came from. They depend
only on the task, not on the model or budget, so there is one per task:

    _index/<stem>[_suffix].jsonl              {"id", "source", "position"} per input row
    _parallel_disambiguation/<stem>.jsonl     {"ambiguous_entities", "unambiguous_entities"}
                                              per input row, taken from the same fields of
                                              data/parallel/<hop>.jsonl; run_entailment_check.py
                                              reads it to build the parallel judge's
                                              disambiguation constraints

An empty compressed_context (a thinking run whose reasoning never closed) is kept as an empty
context. The judge answers "no" for it, so the row counts as a miss and every arm keeps the
same denominator.

The position experiment (``position_<layout>`` stems, see
``src/compression/run_compression_position.py``) is handled as ``parallel``: its 14 layouts
hold the same 400 two-hop questions with the documents reordered, so the chains, the
propositions, the judge and the ambiguous entities are the two-hop ones and only ``context``
differs.

The output tree mirrors outputs/:

    outputs/qwen3/8B/without_thinking/1x/parallel_setting_two_hop.jsonl
 -> entailment_check_input/qwen3/8B/without_thinking/1x/parallel_setting_two_hop.jsonl

Usage
-----
    python src/evaluation/build_entailment_check_input.py           # everything under outputs/
    python src/evaluation/build_entailment_check_input.py --only ablation/gemma4/12B/with_icl/with_thinking
    python src/evaluation/build_entailment_check_input.py --stem 'position_*'
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('run_compression', HERE.parent / 'compression' / 'run_compression.py')
rc = importlib.util.module_from_spec(_spec)
sys.modules['run_compression'] = rc
_spec.loader.exec_module(rc)

# The position experiment's task list lives in its own runner rather than in all_tasks(); its
# stems, input directory and setting names are taken from there so there is one definition.
_pspec = importlib.util.spec_from_file_location(
    'run_compression_position', HERE.parent / 'compression' / 'run_compression_position.py')
rcp = importlib.util.module_from_spec(_pspec)
sys.modules['run_compression_position'] = rcp
_pspec.loader.exec_module(rcp)

REPO = HERE.parents[1]
DEFAULT_OUTPUTS = REPO / 'outputs'
DEFAULT_DEST = REPO / 'entailment_check_input'
# 'fixed' holds the outputs of compressors without length control (RECOMP abstractive is a T5
# model whose output length is decided by generation, with no 0.5x/1x/2x setting). They have a
# single operating point, stored under fixed/ in place of a budget directory.
BUDGET_DIRS = frozenset(rc.BUDGETS) | {'fixed'}
# Fields the compressor never touches: they must match the data file row for row.
CHECKED_FIELDS = ('question', 'logical_chain', 'logical_chain_1', 'logical_chain_2',
                  'key', 'disambiguation_proposition', 'perspectives')


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Per-setting expansion. Each returns {suffix: (input_rows, index_rows)}.
# --------------------------------------------------------------------------- #

def expand_parallel(root_rows, contexts):
    rows, index = [], []
    for src, context in zip(root_rows, contexts):
        for chain_key in ('logical_chain_1', 'logical_chain_2'):
            for pos, prop in enumerate(src[chain_key]):
                rows.append({'context': context, 'proposition': prop})
                index.append({'id': src['id'], 'source': chain_key, 'position': pos})
    return {'': (rows, index)}


def key_position(src) -> int:
    key = src['key'].lower()
    hits = [i for i, prop in enumerate(src['logical_chain']) if key in prop.lower()]
    if len(hits) != 1:
        raise ValueError('%s: key %r matches %d chain elements, expected exactly 1'
                         % (src['id'], src['key'], len(hits)))
    return hits[0]


def expand_conditional(root_rows, contexts):
    plain, plain_idx, keyed, keyed_idx = [], [], [], []
    for src, context in zip(root_rows, contexts):
        k = key_position(src)
        for pos, prop in enumerate(src['logical_chain']):
            ref = {'id': src['id'], 'source': 'logical_chain', 'position': pos}
            if pos == k:
                keyed.append({'context': context, 'key': src['key'],
                              'disambiguation_proposition': src['disambiguation_proposition'],
                              'proposition': prop})
                keyed_idx.append(ref)
            else:
                plain.append({'context': context, 'proposition': prop})
                plain_idx.append(ref)
    return {'_proposition_only': (plain, plain_idx), '_key_proposition': (keyed, keyed_idx)}


def expand_imbalanced(root_rows, contexts):
    rows, index = [], []
    for src, context in zip(root_rows, contexts):
        for pos, perspective in enumerate(src['perspectives']):
            rows.append({'question': src['question'], 'compressed_context': context,
                         'perspective': perspective})
            index.append({'id': src['id'], 'source': 'perspectives', 'position': pos})
    return {'': (rows, index)}


EXPAND = {'parallel': expand_parallel, 'conditional': expand_conditional,
          'imbalanced': expand_imbalanced}


# --------------------------------------------------------------------------- #

def task_table(data_root: Path) -> dict:
    """Output file stem -> (setting, data file). Budgets share both."""
    table = {}
    for t in rc.all_tasks(data_root, Path('/nonexistent'), 'x', 'y'):
        table[t.output_file.stem] = (t.setting, t.input_file)
    table.update(position_table(data_root))
    return table


def position_table(data_root: Path) -> dict:
    """The 14 position-experiment orderings, which ``all_tasks()`` deliberately omits.

    The setting is ``parallel``: same cases, same chains, same propositions, same judge and the
    same ``ambiguous_entities`` -- only the document order inside ``context`` differs. So every
    expansion and check below applies unchanged, and the only thing needed here is the mapping
    from each ``position_<name>`` output stem to its own data file.

    Absent position data is not an error (the experiment is optional); a half-built directory
    is, so once the directory exists all 14 files must be there.
    """
    base = data_root / rcp.INPUT_DIR
    if not base.is_dir():
        return {}
    return {rcp.STEM % name: ('parallel', base / ('%s.jsonl' % name)) for name in rcp.SETTINGS}


def contexts_checked(out_path: Path, root_rows: list[dict]) -> tuple[list[str], int]:
    """compressed_context per row, after checking the file is row-aligned with the data."""
    out_rows = read_jsonl(out_path)
    if len(out_rows) != len(root_rows):
        raise ValueError('%s: %d rows, data has %d' % (out_path, len(out_rows), len(root_rows)))
    contexts, empty = [], 0
    for i, (out, src) in enumerate(zip(out_rows, root_rows)):
        if out.get('id') != src['id']:
            raise ValueError('%s line %d: id %r, data has %r' % (out_path, i, out.get('id'), src['id']))
        for field in CHECKED_FIELDS:
            if field in src and out.get(field) != src[field]:
                raise ValueError('%s line %d: field %r differs from the data' % (out_path, i, field))
        value = out.get('compressed_context')
        if value == rc.ERROR_COMPRESSED_CONTEXT:
            raise ValueError('%s line %d: still holds the error sentinel; retry it first'
                             % (out_path, i))
        if not isinstance(value, str):
            raise ValueError('%s line %d: compressed_context is not a string' % (out_path, i))
        empty += not value.strip()
        contexts.append(value)
    return contexts, empty


def under(rel: Path, prefixes) -> bool:
    return any(str(rel).startswith(prefix.rstrip('/') + '/') for prefix in prefixes)


def output_files(outputs: Path, only, exclude, stems, stem_globs=()) -> list[Path]:
    files = []
    for path in sorted(outputs.rglob('*.jsonl')):
        rel = path.relative_to(outputs)
        if any(part.startswith('_') or '.bak' in part for part in rel.parts):
            continue                    # manifests, raw logs, backups
        if path.stem not in stems or path.parent.name not in BUDGET_DIRS:
            continue
        if stem_globs and not any(fnmatch.fnmatch(path.stem, g) for g in stem_globs):
            continue                    # --stem: one experiment inside a shared budget dir
        if only and not under(rel, only):
            continue
        if exclude and under(rel, exclude):
            continue
        files.append(path)
    return files


def disambiguation_rows(amb, root_rows, label: str) -> list[dict]:
    """One row per proposition, repeating each question's entities across its own chain items."""
    if [a['id'] for a in amb] != [r['id'] for r in root_rows]:
        raise ValueError('ambiguity metadata is not aligned with %s' % label)
    rows = []
    for a, src in zip(amb, root_rows):
        n = len(src['logical_chain_1']) + len(src['logical_chain_2'])
        rows.extend({'ambiguous_entities': a['ambiguous_entities'],
                     'unambiguous_entities': a['unambiguous_entities']} for _ in range(n))
    return rows


def build_disambiguation(data_root: Path, dest: Path, root_cache) -> None:
    amb = {}
    for hop in rc.HOPS:
        stem = 'parallel_setting_%s' % hop
        amb[hop] = root_cache[stem]
        write_jsonl(dest / '_parallel_disambiguation' / (stem + '.jsonl'),
                    disambiguation_rows(amb[hop], root_cache[stem], stem))

    # The position experiment reuses the two-hop entities: the 14 layouts hold the same 400
    # questions in the same order. The file is written once per stem because the judge looks
    # up the disambiguation file by the input file's stem.
    for name in rcp.SETTINGS:
        stem = rcp.STEM % name
        if stem in root_cache:
            write_jsonl(dest / '_parallel_disambiguation' / (stem + '.jsonl'),
                        disambiguation_rows(amb['two_hop'], root_cache[stem], stem))


def parse_args():
    p = argparse.ArgumentParser(description='Expand compression outputs into entailment-check inputs.')
    p.add_argument('--outputs', type=Path, default=DEFAULT_OUTPUTS)
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    p.add_argument('--only', action='append', metavar='REL',
                   help='only process files under this prefix of the outputs directory; '
                        'repeatable, e.g. ablation/gemma4/12B/with_icl/with_thinking')
    p.add_argument('--exclude', action='append', metavar='REL',
                   help='skip files under this prefix of the outputs directory; repeatable')
    p.add_argument('--stem', action='append', metavar='GLOB',
                   help='only process outputs whose file stem matches this glob; repeatable, '
                        'e.g. --stem \'position_*\'. The position experiment and the main 1x '
                        'results share a budget directory, so --only cannot separate them.')
    return p.parse_args()


def main():
    args = parse_args()
    table = task_table(args.data_root)
    root_cache = {stem: read_jsonl(path) for stem, (_, path) in table.items()}

    # Task-level side files: identical for every model and budget.
    for stem, (setting, _) in table.items():
        root_rows = root_cache[stem]
        blank = [''] * len(root_rows)
        for suffix, (_, index) in EXPAND[setting](root_rows, blank).items():
            write_jsonl(args.dest / '_index' / (stem + suffix + '.jsonl'), index)
    build_disambiguation(args.data_root, args.dest, root_cache)

    files = output_files(args.outputs, args.only, args.exclude, table, args.stem or ())
    if not files:
        raise SystemExit('no output files to process')
    total_rows = total_empty = 0
    for path in files:
        setting, _ = table[path.stem]
        root_rows = root_cache[path.stem]
        contexts, empty = contexts_checked(path, root_rows)
        rel = path.relative_to(args.outputs)
        written = []
        for suffix, (rows, _) in EXPAND[setting](root_rows, contexts).items():
            write_jsonl(args.dest / rel.parent / (path.stem + suffix + '.jsonl'), rows)
            written.append(len(rows))
            total_rows += len(rows)
        total_empty += empty
        print('%-72s %4d questions -> %s rows%s' % (rel, len(root_rows), '+'.join(map(str, written)),
                                                    ('  (empty context: %d)' % empty) if empty else ''))
    print('\n%d output files, %d rows written; %d questions with an empty compressed_context'
          % (len(files), total_rows, total_empty))
    print('written to %s' % args.dest)


if __name__ == '__main__':
    main()
