#!/usr/bin/env python3
r"""Build the evidence-position experiment: 14 document orderings of parallel two-hop.

Reads ``data/parallel/two_hop.jsonl`` and ``data/parallel/two_hop.paragraphs.jsonl`` and writes
14 jsonl files plus ``_manifest.json`` to ``data/parallel/position/``.

What the experiment varies
--------------------------
Every case has 22 documents (one case has 20): 4 that carry the evidence and 18 distractors.
The 4 evidence documents form two 2-document blocks, one per logical chain, and a block always
keeps its two hops side by side and in hop order. What moves is (a) how far apart the two
blocks sit and (b) where they sit in the context::

    spread     block gap        front            middle           end
    adjacent   2            0,1 / 2,3        9,10 / 11,12     18,19 / 20,21
    middle     n // 2       0,1 / 11,12      5,6 / 16,17      9,10 / 20,21
    distant    n - 2        -                0,1 / 20,21      -

``distant`` spans the whole context, so its three placements collapse into one and it is
emitted once. Each of the 7 layouts is written twice, ``true1`` and ``true2``, which swap
which chain occupies the earlier block. 3 + 3 + 1 layouts x 2 = 14 files.

The positions above follow from a general formula::

    gap   = {adjacent: 2, middle: n // 2, distant: n - 2}
    span  = gap + 2
    start = {front: 0, middle: (n - span + 1) // 2, end: n - span}
    block A = (start, start + 1)      block B = (start + gap, start + gap + 1)

so the 20-document case scales instead of being special-cased (it lands on 8,9 / 10,11 for
adjacent_middle, 4,5 / 14,15 for middle_middle, and so on).

The distractors keep their released relative order in all 14 files. Only the 4 evidence
documents move, so differences across the 14 files are a position effect and not a shuffling
effect.

Document labels
---------------
By default the documents are renumbered 0..n-1 in display order, which is also the convention
of the released contexts. ``--keep-original-labels`` keeps each document's original number
instead. The four evidence documents are then the only ones out of ascending order, which
reveals them independently of their position; the flag exists to measure that effect. Its
output goes to ``position_original_labels/`` by default.

How the 4 evidence documents are assigned to chains and hops
------------------------------------------------------------
``is_supporting`` marks the 4 evidence paragraphs of each case, but not which hop each one
serves. The assignment of paragraphs to (chain, hop) is recovered by scoring each of the 4
propositions against each of the 4 supporting paragraphs with IDF-weighted token overlap (IDF
over the case's own documents, so entity names dominate) and taking the best of the 4! = 24
one-to-one assignments.

3 cases tie because they hold two near-duplicate supporting documents (``parallel_2hop_0326``
has two ``Zyran Conclave`` tables with identical prefixes), so which duplicate serves which
chain is arbitrary. The tie is broken deterministically by taking the lexicographically
smallest resulting index tuple, and the tied case ids are recorded in the manifest. Since
``true1``/``true2`` counterbalances chain order, an arbitrary choice there cancels across the
pair.

Output
------
``<dest>/<setting>.jsonl`` x 14, each row carrying the released row's fields unchanged
(``id``, ``question``, ``logical_chain_1``, ``logical_chain_2``, ``answer_1``, ``answer_2``,
``flattened_labels``) with a rebuilt ``context``, plus two fields describing the layout:

    position_setting   the file's name
    evidence_slots     presentation indices of (chain1 hop1, chain1 hop2, chain2 hop1, chain2 hop2)

Keeping the released fields unchanged means that ``build_entailment_check_input.py``
(``CHECKED_FIELDS``, the id alignment check, ``expand_parallel``) and the parallel judge's
disambiguation constraints, which come from the two-hop rows, apply unchanged: these files are
parallel two-hop with the documents in a different order, not a new setting.

``<dest>/_manifest.json`` records the flags, the per-layout slots for each context length, the
per-case evidence assignment, and the tied cases.

The output is identical on every run. The scoring sums over a sorted token list, because a set
iterates in an order that depends on the process's string hash seed, and the resulting
last-bit difference is enough to flip the argmax on a tie. ``--verify`` re-derives everything
in a fresh process and compares the result with the existing files line by line, without
writing anything.

The context string is rendered in the released format, ``'Document [%d](Title: %s)%s\n'`` per
paragraph. The script first asserts that this reproduces all 400 released contexts exactly, so
a format change in the data fails here instead of producing contexts that only look right.

Usage
-----
    python src/data/build_position_variants.py                 # write the 14 files
    python src/data/build_position_variants.py --verify        # re-derive and compare, write nothing
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / 'data'
HOP_DIR = 'parallel'
STRUCTURED = 'two_hop.paragraphs.jsonl'
FORMATTED = 'two_hop.jsonl'

# The released row's fields, in the released order. context is rebuilt; the rest pass through.
PASSTHROUGH = ('id', 'question', 'logical_chain_1', 'logical_chain_2',
               'answer_1', 'answer_2', 'flattened_labels')

SPREADS = {'adjacent': lambda n: 2, 'middle': lambda n: n // 2, 'distant': lambda n: n - 2}
PLACEMENTS = {'front': lambda n, span: 0,
              'middle': lambda n, span: (n - span + 1) // 2,
              'end': lambda n, span: n - span}

# Layout order; it fixes the order of SETTINGS and of the files.
LAYOUTS = (('adjacent', 'front'), ('adjacent', 'middle'), ('adjacent', 'end'),
           ('middle', 'front'), ('middle', 'middle'), ('middle', 'end'),
           ('distant', 'middle'))
TOKEN = re.compile(r"[A-Za-z0-9']+")
SCORE_DIGITS = 9          # assignment scores are rounded here before they are compared


def setting_name(spread: str, placement: str, true_index: int) -> str:
    """distant collapses to one placement, so it drops the placement from its name."""
    stem = spread if spread == 'distant' else f'{spread}_{placement}'
    return f'{stem}_true{true_index}'


SETTINGS = tuple(setting_name(s, p, t) for s, p in LAYOUTS for t in (1, 2))


# --------------------------------------------------------------------------- #
# Rendering and reading
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def render(paragraphs, *, renumber: bool) -> str:
    """The released context format. Every block ends in a newline, including the last."""
    out = []
    for position, para in enumerate(paragraphs):
        label = position if renumber else para['idx']
        out.append('Document [%d](Title: %s)%s\n' % (label, para['title'], para['paragraph_text']))
    return ''.join(out)


def check_render_matches_root(structured: list[dict], formatted: list[dict]) -> None:
    if [r['id'] for r in structured] != [r['id'] for r in formatted]:
        raise SystemExit('data: the structured and formatted files are not id-aligned')
    for src, fmt in zip(structured, formatted):
        if render(src['paragraphs'], renumber=False) != fmt['context']:
            raise SystemExit(
                'data: rendering %s from paragraphs does not reproduce its formatted context; '
                'the context format of the data differs from the one this script writes'
                % src['id'])


# --------------------------------------------------------------------------- #
# Which supporting paragraph serves which chain and hop
# --------------------------------------------------------------------------- #

def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(text)}


def assign_evidence(case: dict) -> tuple[list[int], float]:
    """-> ([idx of c1h1, c1h2, c2h1, c2h2], margin over the runner-up assignment)."""
    paragraphs = case['paragraphs']
    supporting = [p for p in paragraphs if p['is_supporting'] in ('chain_1', 'chain_2')]
    propositions = case['logical_chain_1'] + case['logical_chain_2']
    if len(supporting) != 4 or len(propositions) != 4:
        raise SystemExit('%s: expected 4 supporting paragraphs and 4 propositions, got %d and %d'
                         % (case['id'], len(supporting), len(propositions)))

    document_frequency = Counter()
    for para in paragraphs:
        for token in tokens(para['title'] + ' ' + para['paragraph_text']):
            document_frequency[token] += 1
    n = len(paragraphs)
    unseen = math.log(n)
    idf = {token: math.log(n / df) for token, df in document_frequency.items()}

    candidate_tokens = [tokens(p['title'] + ' ' + p['paragraph_text']) for p in supporting]
    # sorted() is not cosmetic: a set iterates in an order that depends on the process's string
    # hash seed, so summing over the raw intersection makes the total differ in the last bits
    # from run to run, which is enough to flip the argmax on a tie. SCORE_DIGITS then collapses
    # whatever float noise is left, so an exact tie is recognised as one and resolved below.
    overlap = [[sum(idf.get(t, unseen) for t in sorted(tokens(prop) & cand))
                for cand in candidate_tokens]
               for prop in propositions]

    # Rank every one-to-one assignment; break ties on the smallest resulting index tuple so
    # that two interchangeable duplicate documents get a fixed, documented order.
    ranked = sorted(
        ((round(sum(overlap[i][perm[i]] for i in range(4)), SCORE_DIGITS),
          tuple(-supporting[perm[i]]['idx'] for i in range(4)), perm)
         for perm in itertools.permutations(range(4))),
        reverse=True)
    best, runner_up = ranked[0], ranked[1]
    return [supporting[best[2][i]]['idx'] for i in range(4)], best[0] - runner_up[0]


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def slots(n: int, spread: str, placement: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Presentation indices of the earlier block and the later block."""
    gap = SPREADS[spread](n)
    span = gap + 2
    start = PLACEMENTS[placement](n, span)
    if start < 0 or start + span > n:
        raise SystemExit('layout %s/%s does not fit %d documents' % (spread, placement, n))
    return (start, start + 1), (start + gap, start + gap + 1)


def build_context(case: dict, evidence: list[int], spread: str, placement: str,
                  true_index: int, *, renumber: bool) -> tuple[str, list[int]]:
    """-> (context, presentation indices of c1h1, c1h2, c2h1, c2h2)."""
    by_idx = {p['idx']: p for p in case['paragraphs']}
    n = len(case['paragraphs'])
    chain1, chain2 = evidence[:2], evidence[2:]
    block_a, block_b = slots(n, spread, placement)
    # true1 puts chain 1 in the earlier block, true2 puts chain 2 there.
    earlier, later = (chain1, chain2) if true_index == 1 else (chain2, chain1)

    placed: dict[int, int] = {}
    for block, chain in ((block_a, earlier), (block_b, later)):
        for slot, idx in zip(block, chain):
            placed[slot] = idx

    # Distractors fill the remaining slots in their released order.
    distractors = [p['idx'] for p in case['paragraphs'] if p['idx'] not in set(evidence)]
    ordered: list[int] = []
    remaining = iter(distractors)
    for slot in range(n):
        ordered.append(placed[slot] if slot in placed else next(remaining))
    if next(remaining, None) is not None:
        raise SystemExit('%s: distractors left over after filling %d slots' % (case['id'], n))
    if sorted(ordered) != sorted(by_idx):
        raise SystemExit('%s: rebuilt order is not a permutation of the documents' % case['id'])

    position_of = {idx: slot for slot, idx in enumerate(ordered)}
    context = render([by_idx[idx] for idx in ordered], renumber=renumber)
    return context, [position_of[idx] for idx in evidence]


def build_rows(structured: list[dict], assignment: dict[str, list[int]], name: str,
               spread: str, placement: str, true_index: int, *, renumber: bool) -> list[dict]:
    rows = []
    for case in structured:
        context, evidence_slots = build_context(
            case, assignment[case['id']], spread, placement, true_index, renumber=renumber)
        row = {key: case[key] for key in PASSTHROUGH if key in case}
        row['context'] = context
        row['position_setting'] = name
        row['evidence_slots'] = evidence_slots
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description='Build the 14 evidence-position variants of parallel two-hop.')
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument('--dest', type=Path, default=None,
                   help='output directory (default: <data-root>/parallel/position, or '
                        '<data-root>/parallel/position_original_labels with '
                        '--keep-original-labels)')
    p.add_argument('--keep-original-labels', action='store_true',
                   help='keep the original document numbers instead of renumbering them '
                        'in display order; the evidence documents are then the only ones '
                        'out of ascending order')
    p.add_argument('--verify', action='store_true', help='re-derive the files and compare them with the existing output; write nothing')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hop = args.data_root / HOP_DIR
    structured = read_jsonl(hop / STRUCTURED)
    formatted = read_jsonl(hop / FORMATTED)
    check_render_matches_root(structured, formatted)
    print('data: %d cases, rendering reproduces every released context' % len(structured))

    renumber = not args.keep_original_labels
    dest = args.dest or (hop / ('position' if renumber else 'position_original_labels'))

    assignment, margins, ties = {}, [], []
    for case in structured:
        evidence, margin = assign_evidence(case)
        assignment[case['id']] = evidence
        margins.append(margin)
        if margin <= 0:
            ties.append(case['id'])
    print('evidence: assigned 4 documents per case; median margin %.1f, %d tied (%s)'
          % (sorted(margins)[len(margins) // 2], len(ties), ', '.join(ties) or 'none'))

    lengths = sorted({len(c['paragraphs']) for c in structured})
    print('layouts (labels %s):' % ('renumbered 0..n-1' if renumber else 'kept from the data'))
    for spread, placement in LAYOUTS:
        line = '  %-16s' % setting_name(spread, placement, 1).removesuffix('_true1')
        for n in lengths:
            a, b = slots(n, spread, placement)
            line += '  n=%d: %s/%s' % (n, ','.join(map(str, a)), ','.join(map(str, b)))
        print(line)

    manifest = {
        'source': {'structured': STRUCTURED, 'formatted': FORMATTED,
                   'cases': len(structured)},
        'renumbered_labels': renumber,
        'settings': list(SETTINGS),
        'slots': {setting_name(s, p, 1).removesuffix('_true1'):
                  {str(n): [list(x) for x in slots(n, s, p)] for n in lengths}
                  for s, p in LAYOUTS},
        'tied_evidence_assignment': ties,
        'evidence_idx': assignment,
    }

    mismatched = []
    for spread, placement in LAYOUTS:
        for true_index in (1, 2):
            name = setting_name(spread, placement, true_index)
            rows = build_rows(structured, assignment, name, spread, placement, true_index,
                              renumber=renumber)
            path = dest / (name + '.jsonl')
            if args.verify:
                if not path.exists():
                    mismatched.append('%s: missing' % name)
                elif read_jsonl(path) != rows:
                    mismatched.append('%s: differs' % name)
            else:
                write_jsonl(path, rows)

    if args.verify:
        existing = dest / '_manifest.json'
        if not existing.exists():
            mismatched.append('_manifest.json: missing')
        elif json.loads(existing.read_text(encoding='utf-8')) != manifest:
            mismatched.append('_manifest.json: differs')
        if mismatched:
            raise SystemExit('verify FAILED\n  ' + '\n  '.join(mismatched))
        print('verify: %d files under %s match a fresh rebuild' % (len(SETTINGS), dest))
        return

    (dest / '_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('wrote %d files + _manifest.json to %s' % (len(SETTINGS), dest))


if __name__ == '__main__':
    main()
