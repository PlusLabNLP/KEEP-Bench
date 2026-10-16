#!/usr/bin/env python3
"""Score the evidence-position experiment: recall metrics x 14 layouts x 3 models.

The judgements are row-aligned with ``entailment_check_input/_index/position_<setting>.jsonl``,
four per case in the fixed order

    (chain 1, hop 1), (chain 1, hop 2), (chain 2, hop 1), (chain 2, hop 2)

A case counts toward a metric when every judgement the metric covers is "yes". Each metric is
the fraction of cases that count:

    All-Rec                   all four judgements
    Para-Rec                  every position i with flattened_labels[i] == 'yes'; for two-hop
                              this is (yes, no, yes, no) in all 400 cases, i.e. the two hop-1
                              propositions, which are the two block-leading ones
    first / second chain      judgements 0 and 1 / judgements 2 and 3
    first / third proposition judgement 0 / judgement 2

Chain index is not a position. ``evidence_slots`` is keyed to (c1h1, c1h2, c2h1, c2h2), and the
counterbalance flips which chain sits in the earlier block: ``true1`` puts chain 1 there,
``true2`` puts chain 2 there. Averaging a chain-keyed column over both halves therefore mixes
the two positions. The block-keyed columns are derived from evidence_slots rather than from the
setting name, and position effects are read off them, not off the chain columns:

    earlier / later block recall        both judgements of the chain in that block
    earlier / later block leading prop  the first judgement of the chain in that block

Every block in all 14 layouts is hop 1 then hop 2, so hop identity and within-block slot are
perfectly collinear. The per-proposition columns are therefore reported as block-leading /
block-trailing and cannot be read as a hop effect.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]

ARMS = (('llama3.1-8b', 'llama3/8B'), ('gemma4-e4b', 'gemma4/E4B'), ('qwen3-8b', 'qwen3/8B'))
DATA_DIR = ROOT / 'data' / 'parallel' / 'position'
SUBDIR = Path('without_thinking') / '1x'
STEM = 'position_%s'

CHAIN_BLOCKS = ((0, 1), (2, 3))          # judgement indices belonging to chain 1 / chain 2
LEADING = (0, 2)                         # the block-leading proposition of each chain


def rows(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_setting(arm_dir, setting):
    """Return (data rows, judgements grouped 4-per-case) with the alignment checked, not assumed."""
    data = rows(DATA_DIR / ('%s.jsonl' % setting))
    index = rows(ROOT / 'entailment_check_input' / '_index' / (STEM % setting + '.jsonl'))
    out = rows(ROOT / 'entailment_check_output' / arm_dir / SUBDIR / (STEM % setting + '.jsonl'))

    if not len(index) == len(out) == 4 * len(data):
        raise SystemExit('%s/%s: %d index, %d judgements, %d cases' %
                         (arm_dir, setting, len(index), len(out), len(data)))
    groups = []
    for case, offset in zip(data, range(0, len(index), 4)):
        window = index[offset:offset + 4]
        expected = [(case['id'], 'logical_chain_%d' % (1 + i // 2), i % 2) for i in range(4)]
        got = [(r['id'], r['source'], r['position']) for r in window]
        if got != expected:
            raise SystemExit('%s/%s: index row %d is %r, expected %r' %
                             (arm_dir, setting, offset, got, expected))
        groups.append([r['judgement'] for r in out[offset:offset + 4]])
    return data, groups


def score(data, groups):
    """Accumulate the six chain-keyed metrics plus their block-keyed counterparts."""
    tally = {k: 0 for k in ('all', 'para', 'chain1', 'chain2', 'prop1', 'prop3',
                            'early_block', 'late_block', 'early_lead', 'late_lead')}
    unknown = 0
    for case, judgements in zip(data, groups):
        yes = [j == 'yes' for j in judgements]
        unknown += sum(j not in ('yes', 'no') for j in judgements)
        labels = case['flattened_labels']

        tally['all'] += all(yes)
        tally['para'] += all(yes[i] for i, lab in enumerate(labels) if lab == 'yes')
        tally['chain1'] += yes[0] and yes[1]
        tally['chain2'] += yes[2] and yes[3]
        tally['prop1'] += yes[0]
        tally['prop3'] += yes[2]

        # Which chain holds the earlier block is read off the slots, so true1/true2 need no
        # special-casing and a renamed setting cannot silently invert the column.
        slots = case['evidence_slots']
        early, late = sorted(CHAIN_BLOCKS, key=lambda pair: slots[pair[0]])
        tally['early_block'] += yes[early[0]] and yes[early[1]]
        tally['late_block'] += yes[late[0]] and yes[late[1]]
        tally['early_lead'] += yes[early[0]]
        tally['late_lead'] += yes[late[0]]
    return tally, unknown


COLUMNS = (('All-Rec', 'all'), ('Para-Rec', 'para'),
           ('first chain recall', 'chain1'), ('second chain recall', 'chain2'),
           ('first proposition recall', 'prop1'), ('third proposition recall', 'prop3'),
           ('earlier block recall', 'early_block'), ('later block recall', 'late_block'),
           ('earlier block leading prop', 'early_lead'), ('later block leading prop', 'late_lead'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(ROOT / 'results' / 'position_14settings.csv'))
    parser.add_argument('--settings', nargs='*', help='subset of layout names (default: all 14)')
    args = parser.parse_args()

    manifest = json.loads((DATA_DIR / '_manifest.json').read_text())
    settings = args.settings or manifest['settings']

    table = []
    unknown_total = 0
    for arm, arm_dir in ARMS:
        for setting in settings:
            data, groups = load_setting(arm_dir, setting)
            tally, unknown = score(data, groups)
            unknown_total += unknown
            row = {'model_setting': arm, 'experiment_setting': setting,
                   'samples': len(data), 'judgements': 4 * len(data)}
            row.update({name: tally[key] / len(data) for name, key in COLUMNS})
            table.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        for row in table:
            writer.writerow({k: ('%.6f' % v if isinstance(v, float) else v)
                             for k, v in row.items()})
    print('%d rows -> %s' % (len(table), out))
    if unknown_total:
        print('warning: %d judgements were neither yes nor no' % unknown_total, file=sys.stderr)


if __name__ == '__main__':
    main()
