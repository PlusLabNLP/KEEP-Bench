#!/usr/bin/env python3
"""Aggregate judge outputs into the case-level metrics reported in the paper.

    python src/evaluation/score.py
    python src/evaluation/score.py --only qwen3/8B --by-subset
    python src/evaluation/score.py --flops results/flops_1x.json --csv results/scores.csv

A run is one directory of judge outputs, e.g. ``entailment_check_output/qwen3/8B/without_thinking/1x``.
Each output file is row-aligned with the input file of the same name, and
``entailment_check_input/_index/<stem>.jsonl`` maps every row to (case id, chain, position).

Case-level metrics (each averaged over the cases of a setting):

    All-Rec   every evidence proposition of the case is preserved
    Para-Rec  parallel: every proposition marked "yes" in flattened_labels is preserved,
              i.e. the propositions that distinguish the two interpretations
    Cond-Rec  conditional: the qualifier is preserved (condition_judgement of the
              qualified proposition)

For a conditional case, All-Rec requires every unqualified proposition, the qualifier
(condition_judgement) and the qualified proposition (overall_judgement). Overall All-Rec is the
mean over all 4,102 cases and is reported only when a run covers every subset.

EPR = kappa * sqrt(A / F), where A is overall All-Rec on a 0-100 scale and F is the average
TFLOPs per case at the default budget (see src/computation/run_flops.py); kappa = 6.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_JUDGEMENTS = REPO / 'entailment_check_output'
DEFAULT_INDEX = REPO / 'entailment_check_input' / '_index'
DEFAULT_DATA = REPO / 'data'
KAPPA = 6.0
DEFAULT_BUDGETS = ('1x', 'fixed')     # budgets at which EPR is defined

HOPS = ('two_hop', 'three_hop', 'four_hop')
SUBSETS = {
    # stem -> (setting, data file used for flattened_labels, or None)
    **{f'parallel_setting_{h}': ('parallel', f'parallel/{h}.jsonl') for h in HOPS},
    **{f'conditional_setting_{h}': ('conditional', None) for h in HOPS},
    'science_domain_three_perspectives': ('science', None),
    'science_domain_four_perspectives': ('science', None),
    'policy_domain': ('policy', None),
}
SETTINGS = ('parallel', 'conditional', 'science', 'policy')
COLUMNS = (('parallel', 'para_rec', 'Parallel Para-Rec'), ('parallel', 'all_rec', 'Parallel All-Rec'),
           ('conditional', 'cond_rec', 'Conditional Cond-Rec'),
           ('conditional', 'all_rec', 'Conditional All-Rec'),
           ('science', 'all_rec', 'Science All-Rec'), ('policy', 'all_rec', 'Policy All-Rec'))


class Incomplete(Exception):
    pass


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def label(row: dict, field: str = 'judgement') -> bool:
    value = row.get(field)
    if value not in ('yes', 'no'):
        raise Incomplete('missing or invalid %s: %r' % (field, row))
    return value == 'yes'


def aligned(run_dir: Path, index_dir: Path, name: str) -> list[tuple[dict, dict]]:
    """(index row, judgement row) pairs for one judge file."""
    out_path, idx_path = run_dir / (name + '.jsonl'), index_dir / (name + '.jsonl')
    if not out_path.exists():
        raise Incomplete('%s is missing' % out_path.name)
    judgements, index = read_jsonl(out_path), read_jsonl(idx_path)
    if len(judgements) != len(index):
        raise Incomplete('%s has %d rows, its index has %d' % (out_path.name, len(judgements), len(index)))
    return list(zip(index, judgements))


def score_parallel(run_dir, index_dir, stem, data_file):
    flat = {r['id']: r['flattened_labels'] for r in read_jsonl(data_file)}
    chain_len = {r['id']: len(r['logical_chain_1']) for r in read_jsonl(data_file)}
    cases: dict[str, dict[int, bool]] = {}
    for ref, row in aligned(run_dir, index_dir, stem):
        offset = 0 if ref['source'] == 'logical_chain_1' else chain_len[ref['id']]
        cases.setdefault(ref['id'], {})[offset + ref['position']] = label(row)
    results = {}
    for case_id, z in cases.items():
        labels = flat[case_id]
        if sorted(z) != list(range(len(labels))):
            raise Incomplete('%s: case %s has judgements for positions %s' % (stem, case_id, sorted(z)))
        results[case_id] = {'all_rec': all(z.values()),
                            'para_rec': all(z[i] for i, f in enumerate(labels) if f == 'yes')}
    return results


def score_conditional(run_dir, index_dir, stem):
    cases: dict[str, list[bool]] = {}
    cond: dict[str, bool] = {}
    for ref, row in aligned(run_dir, index_dir, stem + '_proposition_only'):
        cases.setdefault(ref['id'], []).append(label(row))
    for ref, row in aligned(run_dir, index_dir, stem + '_key_proposition'):
        condition = label(row, 'condition_judgement')
        cond[ref['id']] = condition
        cases.setdefault(ref['id'], []).extend([condition, label(row, 'overall_judgement')])
    if set(cond) != set(cases):
        raise Incomplete('%s: cases without a qualified proposition' % stem)
    return {case_id: {'all_rec': all(z), 'cond_rec': cond[case_id]} for case_id, z in cases.items()}


def score_perspectives(run_dir, index_dir, stem):
    cases: dict[str, list[bool]] = {}
    for ref, row in aligned(run_dir, index_dir, stem):
        cases.setdefault(ref['id'], []).append(label(row))
    return {case_id: {'all_rec': all(z)} for case_id, z in cases.items()}


def score_run(run_dir: Path, index_dir: Path, data_root: Path, verbose: bool):
    """-> {setting: {case_id: metrics}}, {stem: {case_id: metrics}}, [skipped stems]"""
    by_setting: dict[str, dict] = {s: {} for s in SETTINGS}
    by_subset, skipped = {}, []
    for stem, (setting, data_file) in SUBSETS.items():
        try:
            if setting == 'parallel':
                result = score_parallel(run_dir, index_dir, stem, data_root / data_file)
            elif setting == 'conditional':
                result = score_conditional(run_dir, index_dir, stem)
            else:
                result = score_perspectives(run_dir, index_dir, stem)
        except Incomplete as error:
            skipped.append(stem)
            if verbose:
                print('  skip %s: %s' % (stem, error), file=sys.stderr)
            continue
        by_subset[stem] = result
        by_setting[setting].update(result)
    return by_setting, by_subset, skipped


def mean(values) -> float:
    values = list(values)
    return 100.0 * sum(values) / len(values) if values else float('nan')


def find_runs(root: Path, only) -> list[Path]:
    runs = sorted({p.parent for p in root.rglob('*.jsonl')
                   if p.stem.removesuffix('_proposition_only').removesuffix('_key_proposition') in SUBSETS})
    if only:
        runs = [r for r in runs if any(str(r.relative_to(root)).startswith(o.rstrip('/')) for o in only)]
    return runs


def load_flops(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return {row['method']: float(row['tflops_per_case']) for row in data}


def main() -> None:
    p = argparse.ArgumentParser(description='Aggregate judge outputs into All-Rec, Para-Rec, Cond-Rec and EPR.')
    p.add_argument('--judgements', type=Path, default=DEFAULT_JUDGEMENTS)
    p.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--only', action='append', metavar='PREFIX',
                   help='score only runs under this prefix of --judgements (repeatable)')
    p.add_argument('--by-subset', action='store_true', help='also print All-Rec per subset')
    p.add_argument('--flops', type=Path, default=None,
                   help='JSON list of {"method", "tflops_per_case"}; adds EPR for default-budget runs')
    p.add_argument('--csv', type=Path, default=None, help='write the table to this file')
    p.add_argument('--verbose', action='store_true', help='explain skipped subsets')
    args = p.parse_args()

    flops = load_flops(args.flops)
    runs = find_runs(args.judgements, args.only)
    if not runs:
        raise SystemExit('no judge outputs found under %s' % args.judgements)

    header = ['run'] + [c[2] for c in COLUMNS] + ['Overall All-Rec', 'EPR']
    table = []
    for run_dir in runs:
        rel = run_dir.relative_to(args.judgements)
        by_setting, by_subset, skipped = score_run(run_dir, args.index, args.data_root, args.verbose)
        row = {'run': str(rel)}
        for setting, metric, name in COLUMNS:
            cases = by_setting[setting]
            row[name] = mean(m[metric] for m in cases.values()) if cases else float('nan')
        overall = mean(m['all_rec'] for s in SETTINGS for m in by_setting[s].values())
        row['Overall All-Rec'] = overall if not skipped else float('nan')
        method = str(rel.parent.parent)              # e.g. qwen3/8B from qwen3/8B/without_thinking/1x
        row['EPR'] = float('nan')
        if method in flops and rel.name in DEFAULT_BUDGETS and not skipped:
            row['EPR'] = KAPPA * math.sqrt(overall / flops[method])
        table.append(row)
        if skipped:
            row['run'] += '  (incomplete: %d of %d subsets missing)' % (len(skipped), len(SUBSETS))
        if args.by_subset:
            for stem, cases in by_subset.items():
                print('%-60s %-36s n=%4d  All-Rec %.1f'
                      % (rel, stem, len(cases), mean(m['all_rec'] for m in cases.values())))

    width = max(len(r['run']) for r in table)
    print(' '.join([header[0].ljust(width)] + ['%22s' % h for h in header[1:]]))
    for r in table:
        cells = ['%22s' % ('-' if math.isnan(r[h]) else '%.1f' % r[h]) if h != 'EPR'
                 else '%22s' % ('-' if math.isnan(r[h]) else '%.3f' % r[h]) for h in header[1:]]
        print(' '.join([r['run'].ljust(width)] + cells))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in table:
                writer.writerow({h: ('' if isinstance(r[h], float) and math.isnan(r[h]) else
                                     (round(r[h], 3) if isinstance(r[h], float) else r[h])) for h in header})
        print('wrote %s' % args.csv)


if __name__ == '__main__':
    main()
