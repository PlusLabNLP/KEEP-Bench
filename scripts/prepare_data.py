#!/usr/bin/env python3
"""Download KEEP-Bench from Hugging Face and write the layout the scripts in src/ read.

    python scripts/prepare_data.py                      # from the Hub (requires `hf auth login`)
    python scripts/prepare_data.py --local /path/to/KEEP-Bench   # from a local copy of the dataset repo

The dataset has one config per setting. Each config is split into its subsets and written
twice: ``<subset>.jsonl`` holds every field except ``paragraphs`` (``context`` is the input
given to compressors), and ``<subset>.paragraphs.jsonl`` holds every field except ``context``
(the structured paragraphs used by sentence-level compressors and the position experiment).

    data/parallel/{two_hop,three_hop,four_hop}.jsonl
    data/conditional/{two_hop,three_hop,four_hop}.jsonl
    data/science/{three_perspectives,four_perspectives}.jsonl
    data/policy/four_perspectives.jsonl

Rows keep the order of the dataset. The ICL demonstrations in data/icl/ are part of this
repository and are not touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / 'data'
HF_REPO = 'uclanlp/KEEP-Bench'

HOPS = {2: 'two_hop', 3: 'three_hop', 4: 'four_hop'}
PERSPECTIVES = {3: 'three_perspectives', 4: 'four_perspectives'}

# config -> (output directory, function giving a row's subset, expected rows per subset)
CONFIGS = {
    'parallel': ('parallel', lambda r: HOPS[r['hops']],
                 {'two_hop': 400, 'three_hop': 300, 'four_hop': 200}),
    'conditional': ('conditional', lambda r: HOPS[r['hops']],
                    {'two_hop': 604, 'three_hop': 493, 'four_hop': 267}),
    'imbalanced_science': ('science', lambda r: PERSPECTIVES[len(r['perspectives'])],
                           {'three_perspectives': 154, 'four_perspectives': 856}),
    'imbalanced_policy': ('policy', lambda r: PERSPECTIVES[len(r['perspectives'])],
                          {'four_perspectives': 828}),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def config_file(config: str, args) -> Path:
    relative = f'data/{config}/test.jsonl'
    if args.local:
        return args.local / relative
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(args.repo, relative, repo_type='dataset', revision=args.revision))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--repo', default=HF_REPO, help='Hugging Face dataset repository')
    p.add_argument('--revision', default=None, help='dataset revision (branch, tag or commit)')
    p.add_argument('--local', type=Path, default=None,
                   help='read data/<config>/test.jsonl from this directory instead of the Hub')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    total = 0
    for config, (directory, subset_of, expected) in CONFIGS.items():
        rows = read_jsonl(config_file(config, args))
        subsets: dict[str, list[dict]] = {}
        for row in rows:
            subsets.setdefault(subset_of(row), []).append(row)
        counts = {name: len(subset) for name, subset in subsets.items()}
        if counts != expected:
            raise SystemExit(f'{config}: expected subset sizes {expected}, got {counts}')
        for name, subset in subsets.items():
            base = args.out / directory / name
            write_jsonl(base.with_suffix('.jsonl'),
                        [{k: v for k, v in r.items() if k != 'paragraphs'} for r in subset])
            write_jsonl(base.with_name(name + '.paragraphs.jsonl'),
                        [{k: v for k, v in r.items() if k != 'context'} for r in subset])
            print(f'{directory}/{name}: {len(subset)} cases')
        total += len(rows)
    print(f'wrote {total} cases to {args.out}')


if __name__ == '__main__':
    main()
