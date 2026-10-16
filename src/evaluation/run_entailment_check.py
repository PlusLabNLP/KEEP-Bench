#!/usr/bin/env python3
"""Run the entailment judges over entailment_check_input/ and write entailment_check_output/.

Each input file is routed to one of five judges by its file name:

    judge          input files                                        prompt
    parallel       parallel_setting_<hop>.jsonl, position_*.jsonl     PROPOSITION_PROMPT
                                                                      + DISAMBIGUATION_BLOCK
    proposition    conditional_setting_<hop>_proposition_only.jsonl   PROPOSITION_PROMPT
    key            conditional_setting_<hop>_key_proposition.jsonl    KEY_PROMPT
    perspective    science_domain_*.jsonl                             PERSPECTIVE_PROMPT
    jurisdiction   policy_domain.jsonl                                JURISDICTION_PROMPT

The parallel judge appends one disambiguation constraint per ambiguous entity of the question.
The entities come from ``_parallel_disambiguation/<stem>.jsonl``, which
``build_entailment_check_input.py`` writes row-aligned with each parallel input file. The
``position_*`` files hold the parallel two-hop cases with the documents reordered, so they use
the parallel judge unchanged.

Every row is one call to the OpenAI Responses API,
``client.responses.create(model=..., input=messages, reasoning={"effort": "high"})``, with
gpt-5.2 as the judge (``--model``, ``--reasoning-effort``). The messages are a system turn and
a user turn that holds the prompt, then ``#### Input`` and the row as JSON, then ``#### Output``.

Output
------
One row per input row, in the same order, at the same relative path under the output directory:

    {"judgement": "yes"|"no"}                                   parallel, proposition,
                                                                perspective, jurisdiction
    {"condition_judgement": "yes"|"no",
     "overall_judgement": "yes"|"no"}                           key

A request that raises or returns an unparseable reply is retried up to ``--max-retries`` times
with exponential backoff. A row that still fails is written as ``{"error": ...}`` and the file
is reported as incomplete; an error row is never counted as "no". ``--retry-errors`` re-requests
only the error rows of an existing output file.

Two kinds of row are answered "no" without a call, with a ``"note"`` field:

    empty context      the context is empty, e.g. a thinking run whose reasoning never closed;
                       note "empty context"
    placeholder        perspective and jurisdiction judges only: the context is exactly the
                       compressor's "This passage doesn't contain relevant information to the
                       question." reply; note "irrelevant-context placeholder"

Runs are resumable. A file whose output has as many rows as the input and no error row is
skipped, so an interrupted run is continued by starting it again. A file with an incomplete
output is skipped unless ``--retry-errors`` is given. Output files are written atomically.

A full run is expensive: it issues one high-effort reasoning call per proposition (or
perspective). ``--dry-run`` counts the requests and prints one complete prompt per judge
without calling the API; ``--only`` restricts the run to one slice.

Usage
-----
    export OPENAI_API_KEY=...
    python src/evaluation/run_entailment_check.py --dry-run
    python src/evaluation/run_entailment_check.py --only qwen3/8B/without_thinking/1x
    python src/evaluation/run_entailment_check.py                    # every file not yet done
    python src/evaluation/run_entailment_check.py --retry-errors     # re-request failed rows
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / 'entailment_check_input'
DEFAULT_OUTPUT = REPO / 'entailment_check_output'
DEFAULT_MODEL = 'gpt-5.2'
DEFAULT_MAX_CONCURRENCY = 256
DEFAULT_MAX_RETRIES = 5
# Per-request timeout. The SDK default is 600 s, and files are processed one after another,
# so a single hung request stalls the whole run until it times out. Typical latency is a few
# seconds. A timeout raises APITimeoutError, which the retry loop in judge_row handles.
DEFAULT_TIMEOUT_S = 120.0

# USD per 1M tokens (GPT-5.2 API pricing), used for the per-request usage log. Other models
# report token usage without a cost estimate.
MODEL_PRICING_USD_PER_MILLION = {
    'gpt-5.2': {'input': 1.75, 'cached_input': 0.175, 'output': 14.0},
}

# --------------------------------------------------------------------------- #
# Judge prompts, as used for the paper's results. Editing them changes the scores.
# --------------------------------------------------------------------------- #

JSON_EDITOR = 'You are a precise JSON editor. Output only JSON that conforms to the schema.'
JSON_JUDGE = 'You are a precise JSON judge. Output only JSON that conforms to the schema.'

PROPOSITION_PROMPT = """
You are given a JSON input with two keys: "context" and "proposition".

Your task is to determine whether the proposition is entailed by the context.

Entailment means that the proposition can be logically inferred from at least one coherent part of the context.

Important clarifications:
- The context may contain irrelevant, noisy, or even contradictory information. This does NOT invalidate entailment.
  As long as there exists at least one subset of the context from which the proposition can be reasonably inferred,
  the proposition should be considered entailed.
- The proposition does not need to match the wording used in the context.
  As long as the proposition holds under a reasonable interpretation of some part of the context,
  it should be considered entailed.

If the context entails the proposition, output exactly:
{"judgement":"yes"}

Otherwise, output exactly:
{"judgement":"no"}

Do not include any explanation or extra text. Output JSON only.

"""

DISAMBIGUATION_BLOCK = """

Additional Disambiguation Constraints (all must be applied):
{constraints_json}

Rules for judgement (apply to every constraint listed above):
- Each listed "unambiguous entity" consists of the ambiguous name plus a distinguishing qualifier (for example a country, a county, a date, a type, or an associated person or work). The constraint is about identifying the correct entity. It does not require the label's exact wording or its parenthetical form, and it does not require every part of the qualifier to be reproduced, as long as what appears is enough to single out that entity rather than the other listed alternatives.
- If the proposition mentions one of the listed "unambiguous entities", then context can be judged as entailment only if the proposition-relevant statement in context is tied to that same entity, because that entity's distinguishing qualifier, or an unambiguous equivalent of it, appears in or is attached to that statement.
- If the proposition-relevant statement carries only the bare ambiguous name, with no distinguishing qualifier attached to it, the judgement must be "no", even if that qualifier appears elsewhere in the context in connection with something else.
- If the proposition-relevant statement is tied to a different listed entity than the one the proposition mentions, the judgement must be "no" even if other content appears compatible.
- The context may instead keep the alternatives apart, attaching each one to its own distinguishing qualifier (for example "X is a town in County A, and X is also a place in County B"). This satisfies the constraint for whichever alternative the proposition mentions, provided that alternative's facts are not attributed to the other.
- If several constraints are listed, the judgement must be "no" unless every constraint whose entities the proposition mentions is satisfied.
- These are necessary conditions only; all original entailment requirements must still be satisfied.
"""

KEY_PROMPT = """
You are given a JSON input with four keys: "context", "key", "disambiguation_proposition", and "proposition".

Your task is to make two entailment judgements.

The relationship among "disambiguation_proposition", "key", and "proposition" is important:
- "disambiguation_proposition" describes the broader ambiguous set in which the condition from "key" selects a subset or imposes a constraint.
- Use "disambiguation_proposition" to understand what kind of ambiguity the key is resolving and what collection of possible cases the key is restricting.
- This helps you judge whether the context entails the specific condition in "key".
- "key" is a specific condition or scope restriction.
- "proposition" is a claim that must be interpreted under that specific condition.
- In other words, the proposition is not free-standing. Its later content must be understood as being restricted by the condition in "key".

1. First, determine whether the condition in "key" is entailed by the context.
   Write this as "condition_judgement".
   To judge this, check whether the context contains or establishes the specific conditioned scope expressed by the key.
   For example:
   - If the key is "For the Lyra surnamed Voss", then the context should contain or establish that specific Lyra surnamed Voss.
   - If the key is "For the original mechanism", then the context should contain or establish that specific original mechanism.
   If the context does not establish the condition itself, then "condition_judgement" should be "no".

2. Then determine whether the proposition is entailed by the context, but this second judgement must respect the condition:
   - If the context does not entail the condition in "key", then "overall_judgement" must be "no".
   - If the context does entail the condition in "key", then judge whether the full proposition is entailed by the context under that condition.
   - The proposition should count as entailed only if the context supports the later information in the proposition as being about the scope restricted by the key.
   - In other words, the context must not merely support a similar statement somewhere else; it must support that statement specifically under the condition introduced by the key.

Entailment means that the statement can be logically inferred from at least one coherent part of the context.

Important clarifications:
- The context may contain irrelevant, noisy, or even contradictory information. This does NOT invalidate entailment.
  As long as there exists at least one subset of the context from which the target statement can be reasonably inferred,
  it should be considered entailed.
- The key and proposition do not need to match the wording used in the context.
  As long as the context expresses the same meaning, they should be considered entailed.

Output exactly one JSON object in the following format:
{"condition_judgement":"yes" or "no","overall_judgement":"yes" or "no"}

Do not include any explanation or extra text. Output JSON only.
"""

PERSPECTIVE_PROMPT = """
You are given a JSON input with three keys: "question", "compressed_context", and "perspective".

Your task is to determine whether the compressed_context entails the perspective as an answer to the question.

Judge entailment by meaning, not by exact wording.
The compressed_context does NOT need to restate every word or every minor detail of the perspective.
However, it must clearly express the same answer in substance.

Important clarifications:
- The compressed_context may contain multiple different perspectives or answer candidates. This is allowed.
- If any coherent part of the compressed_context points to the same underlying answer to the question as the perspective, judge "yes".
- The judgement is about whether the perspective itself is entailed, not whether a broader topic related to it is mentioned.
- A more general category does NOT count unless it clearly conveys the same answer as the perspective.
- Missing minor wording details can be acceptable, but some coherent part of the compressed_context must still point to the same underlying answer to the question.
- If the compressed_context merely mentions a broader family that could include the perspective, but does not itself point to the same underlying answer, judge "no".
- If some coherent part of the compressed_context clearly conveys the same answer as the perspective, even with paraphrasing, judge "yes".

If the compressed_context entails the perspective, output exactly:
{"judgement":"yes"}

Otherwise, output exactly:
{"judgement":"no"}

Do not include any explanation or extra text. Output JSON only.
"""

JURISDICTION_PROMPT = """
You are given a JSON input with three keys: "question", "compressed_context", and "perspective".

Your task is to determine whether the compressed_context entails the perspective as an answer to the question.

Judge entailment by meaning, not by exact wording.
The compressed_context does NOT need to restate every word or every minor detail of the perspective.
However, it must clearly express the same answer in substance.

Important clarifications:
- The compressed_context may contain multiple different perspectives or answer candidates. This is allowed.
- If any coherent part of the compressed_context points to the same underlying answer to the question as the perspective, judge "yes".
- The judgement is about whether the perspective itself is entailed, not whether a broader topic related to it is mentioned.
- The jurisdiction matters. The compressed_context must clearly indicate the country, region, legal system, or jurisdiction to which the entailed answer applies.
- If the compressed_context states the substantive rule but does not identify the applicable jurisdiction, judge "no".
- The jurisdiction does not need to be named with the exact same wording as the perspective. An unambiguous equivalent name, abbreviation, governing body, agency, or legal framework can count.
- A more general category does NOT count unless it clearly conveys the same answer as the perspective.
- Missing minor wording details can be acceptable, but some coherent part of the compressed_context must still point to the same underlying answer to the question.
- If the compressed_context merely mentions a broader family that could include the perspective, but does not itself point to the same underlying answer, judge "no".
- If some coherent part of the compressed_context clearly conveys the same answer as the perspective, even with paraphrasing, judge "yes".

If the compressed_context entails the perspective, output exactly:
{"judgement":"yes"}

Otherwise, output exactly:
{"judgement":"no"}

Do not include any explanation or extra text. Output JSON only.
"""

IRRELEVANT_CONTEXT_PLACEHOLDER = "This passage doesn't contain relevant information to the question."


# --------------------------------------------------------------------------- #
# Message builders: system turn + "{prompt}\n#### Input\n{json}\n#### Output\n"
# --------------------------------------------------------------------------- #

def messages(system: str, prompt: str, payload: dict) -> list[dict]:
    user_json = json.dumps(payload, ensure_ascii=False)
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': f'{prompt}\n#### Input\n{user_json}\n#### Output\n'}]


def disambiguation_block(row: dict) -> str:
    """One constraint per (ambiguous entity, unambiguous entities) pair; empty if there are none."""
    pairs = []
    for amb, options in zip(row.get('ambiguous_entities') or [], row.get('unambiguous_entities') or []):
        if not isinstance(amb, str) or not amb.strip() or not isinstance(options, list):
            continue
        options = [u for u in options if isinstance(u, str) and u.strip()]
        if options:
            pairs.append({'ambiguous entity': amb.strip(), 'unambiguous entities': options})
    if not pairs:
        return ''
    return DISAMBIGUATION_BLOCK.format(constraints_json=json.dumps(pairs, ensure_ascii=False))


class Judge:
    """What one kind of input file needs: how to build a request, how to read the reply."""

    def __init__(self, name, fields, build, parse, context_field):
        self.name, self.fields, self.build, self.parse = name, fields, build, parse
        self.context_field = context_field


def parse_single(parsed: dict):
    value = str(parsed.get('judgement', '')).strip().lower()
    return {'judgement': value} if value in {'yes', 'no'} else None


def parse_dual(parsed: dict):
    cond = str(parsed.get('condition_judgement', '')).strip().lower()
    overall = str(parsed.get('overall_judgement', '')).strip().lower()
    if cond in {'yes', 'no'} and overall in {'yes', 'no'}:
        return {'condition_judgement': cond, 'overall_judgement': overall}
    return None


def judge_for(stem: str) -> Judge:
    # position_* is the evidence-position experiment: the same two-hop cases with the documents
    # reordered. It uses the parallel judge unchanged so that its judgements are comparable
    # with the parallel ones.
    if stem.startswith('parallel_setting_') or stem.startswith('position_'):
        return Judge('parallel', ('context', 'proposition'),
                     lambda e, extra: messages(JSON_EDITOR, PROPOSITION_PROMPT + disambiguation_block(extra), e),
                     parse_single, 'context')
    if stem.startswith('conditional_setting_') and stem.endswith('_proposition_only'):
        return Judge('proposition', ('context', 'proposition'),
                     lambda e, extra: messages(JSON_EDITOR, PROPOSITION_PROMPT, e),
                     parse_single, 'context')
    if stem.startswith('conditional_setting_') and stem.endswith('_key_proposition'):
        return Judge('key', ('context', 'key', 'disambiguation_proposition', 'proposition'),
                     lambda e, extra: messages(JSON_EDITOR, KEY_PROMPT, e),
                     parse_dual, 'context')
    if stem.startswith('science_domain_'):
        return Judge('perspective', ('question', 'compressed_context', 'perspective'),
                     lambda e, extra: messages(JSON_JUDGE, PERSPECTIVE_PROMPT, e),
                     parse_single, 'compressed_context')
    if stem == 'policy_domain':
        return Judge('jurisdiction', ('question', 'compressed_context', 'perspective'),
                     lambda e, extra: messages(JSON_JUDGE, JURISDICTION_PROMPT, e),
                     parse_single, 'compressed_context')
    raise ValueError('unrecognized input file: %s' % stem)


def try_parse_json(txt: str) -> dict:
    """Drop a </think> prefix, parse as JSON, and fall back to the outermost {...}."""
    if '</think>' in txt:
        txt = txt.split('</think>', 1)[1]
    txt = txt.strip()
    try:
        value = json.loads(txt)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    start, end = txt.find('{'), txt.rfind('}')
    if start == -1 or end == -1:
        return {}
    cleaned = re.sub(r'\s+', ' ', txt[start:end + 1].replace('\n', ' ').replace('\r', ' '))
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def shortcut(judge: Judge, entry: dict):
    """A verdict that needs no judge call, or None."""
    context = str(entry.get(judge.context_field, '')).strip()
    no = ({'condition_judgement': 'no', 'overall_judgement': 'no'} if judge.name == 'key'
          else {'judgement': 'no'})
    if not context:
        return {**no, 'note': 'empty context'}
    if judge.name in {'perspective', 'jurisdiction'} and context == IRRELEVANT_CONTEXT_PLACEHOLDER:
        return {**no, 'note': 'irrelevant-context placeholder'}
    return None


# --------------------------------------------------------------------------- #

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


def is_error(row) -> bool:
    return not isinstance(row, dict) or 'error' in row


def input_files(root: Path, only, exclude) -> list[Path]:
    def under(rel, prefixes):
        return any(str(rel).startswith(p.rstrip('/') + '/') or str(rel) == p for p in prefixes)
    files = []
    for path in sorted(root.rglob('*.jsonl')):
        rel = path.relative_to(root)
        if rel.parts[0].startswith('_'):
            continue                                    # _index, _parallel_disambiguation
        if only and not under(rel, only):
            continue
        if exclude and under(rel, exclude):
            continue
        files.append(path)
    return files


def report_usage(response, model: str, label: str) -> None:
    usage = getattr(response, 'usage', None)
    if usage is None:
        tqdm.write('[usage] %s model=%s usage unavailable' % (label, model))
        return

    input_tokens = int(getattr(usage, 'input_tokens', 0) or 0)
    output_tokens = int(getattr(usage, 'output_tokens', 0) or 0)
    input_details = getattr(usage, 'input_tokens_details', None)
    cached_tokens = int(getattr(input_details, 'cached_tokens', 0) or 0)
    uncached_tokens = max(0, input_tokens - cached_tokens)

    pricing = MODEL_PRICING_USD_PER_MILLION.get(model)
    if pricing is None:
        cost_text = 'estimated_cost=unavailable'
    else:
        cost = (uncached_tokens * pricing['input']
                + cached_tokens * pricing['cached_input']
                + output_tokens * pricing['output']) / 1_000_000
        cost_text = 'estimated_cost=$%.8f' % cost

    tqdm.write('[usage] %s model=%s input=%d cached_input=%d output=%d %s' %
               (label, model, input_tokens, cached_tokens, output_tokens, cost_text))


async def judge_row(client, args, judge, entry, extra, semaphore, label):
    request = judge.build({f: entry[f] for f in judge.fields}, extra)
    last = None
    async with semaphore:
        for attempt in range(args.max_retries + 1):
            try:
                response = await client.responses.create(
                    model=args.model, input=request, reasoning={'effort': args.reasoning_effort})
                report_usage(response, args.model, label)
                raw = (response.output_text or '').strip()
                verdict = judge.parse(try_parse_json(raw))
                if verdict is not None:
                    return verdict
                last = 'unparseable reply: %r' % raw[:200]
            except Exception as error:                  # noqa: BLE001 -- recorded, not swallowed
                last = '%s: %s' % (type(error).__name__, error)
            if attempt < args.max_retries:
                await asyncio.sleep(min(2 ** attempt, 60) + random.random())
    return {'error': last}


async def run_file(client, args, in_path: Path, out_path: Path, semaphore) -> tuple[int, int, int]:
    """Judge one input file. Returns (rows sent, rows shortcut, rows still failed)."""
    judge = judge_for(in_path.stem)
    entries = read_jsonl(in_path)
    for i, entry in enumerate(entries):
        missing = [f for f in judge.fields if f not in entry]
        if missing:
            raise ValueError('%s line %d: missing fields %s' % (in_path, i, missing))

    extras = [None] * len(entries)
    if judge.name == 'parallel':
        extras = read_jsonl(args.input / '_parallel_disambiguation' / (in_path.stem + '.jsonl'))
        if len(extras) != len(entries):
            raise ValueError('%s: disambiguation file has %d rows, input has %d'
                             % (in_path, len(extras), len(entries)))

    results = [None] * len(entries)
    if out_path.exists() and args.retry_errors:
        previous = read_jsonl(out_path)
        if len(previous) == len(entries):
            results = [None if is_error(r) else r for r in previous]

    todo, shortcuts = [], 0
    for i, entry in enumerate(entries):
        if results[i] is not None:
            continue
        verdict = shortcut(judge, entry)
        if verdict is not None:
            results[i] = verdict
            shortcuts += 1
        else:
            todo.append(i)

    async def one(i):
        label = '%s:%d' % (in_path.relative_to(args.input), i + 1)
        return i, await judge_row(client, args, judge, entries[i], extras[i], semaphore, label)

    coros = [one(i) for i in todo]
    for coro in tqdm(asyncio.as_completed(coros), total=len(coros),
                     desc=str(in_path.relative_to(args.input)), leave=False):
        i, verdict = await coro
        results[i] = verdict
    write_jsonl(out_path, results)
    return len(todo), shortcuts, sum(is_error(r) for r in results)


def done(in_path: Path, out_path: Path) -> bool:
    if not out_path.exists():
        return False
    rows = read_jsonl(out_path)
    return len(rows) == sum(1 for line in in_path.open(encoding='utf-8') if line.strip()) \
        and not any(is_error(r) for r in rows)


def parse_args():
    p = argparse.ArgumentParser(description='Run the entailment judges over entailment_check_input.')
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--reasoning-effort', default='high', help='judge reasoning effort (the paper uses high)')
    p.add_argument('--max-concurrency', type=int, default=DEFAULT_MAX_CONCURRENCY)
    p.add_argument('--max-retries', type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT_S,
                   metavar='SECONDS', help='per-request timeout (default: %(default)ss)')
    p.add_argument('--only', action='append', metavar='REL',
                   help='only judge files under this prefix of the input directory; repeatable, '
                        'e.g. qwen3/8B/without_thinking/1x')
    p.add_argument('--exclude', action='append', metavar='REL', help='skip files under this prefix; repeatable')
    p.add_argument('--retry-errors', action='store_true',
                   help='for existing output files, re-request only the error rows')
    p.add_argument('--dry-run', action='store_true',
                   help='count the requests and print one full prompt per judge, '
                        'without calling the API')
    return p.parse_args()


def dry_run(args, files):
    per_judge, shown = {}, set()
    for path in files:
        judge = judge_for(path.stem)
        entries = read_jsonl(path)
        extras = (read_jsonl(args.input / '_parallel_disambiguation' / (path.stem + '.jsonl'))
                  if judge.name == 'parallel' else [None] * len(entries))
        out_path = args.output / path.relative_to(args.input)
        state = 'done' if done(path, out_path) else 'todo'
        send = sum(shortcut(judge, e) is None for e in entries)
        stats = per_judge.setdefault(judge.name, {'files': 0, 'rows': 0, 'send': 0, 'done_files': 0})
        stats['files'] += 1
        stats['rows'] += len(entries)
        stats['send'] += send if state == 'todo' else 0
        stats['done_files'] += state == 'done'
        if judge.name not in shown:
            shown.add(judge.name)
            i = next(k for k, e in enumerate(entries) if shortcut(judge, e) is None)
            request = judge.build({f: entries[i][f] for f in judge.fields}, extras[i])
            print('=' * 100)
            print('[%s]  %s  line %d' % (judge.name, path.relative_to(args.input), i))
            print('--- system ---\n' + request[0]['content'])
            print('--- user ---\n' + request[1]['content'])
    print('=' * 100)
    total = 0
    for name, s in per_judge.items():
        print('%-13s %4d files (%d done)  %7d rows  %7d requests to send' %
              (name, s['files'], s['done_files'], s['rows'], s['send']))
        total += s['send']
    print('total requests to send: %d (model=%s, reasoning=%s)'
          % (total, args.model, args.reasoning_effort))


async def main():
    args = parse_args()
    files = input_files(args.input, args.only, args.exclude)
    if not files:
        raise SystemExit('no input files found')
    if args.dry_run:
        return dry_run(args, files)
    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is not set')

    client = AsyncOpenAI(max_retries=0, timeout=args.timeout)   # retries are ours, see judge_row
    semaphore = asyncio.Semaphore(args.max_concurrency)
    incomplete = []
    try:
        for n, in_path in enumerate(files, 1):
            rel = in_path.relative_to(args.input)
            out_path = args.output / rel
            if done(in_path, out_path):
                continue
            if out_path.exists() and not args.retry_errors:
                print('skipping %s: output exists but is incomplete; rerun with --retry-errors' % rel)
                incomplete.append(rel)
                continue
            sent, cut, failed = await run_file(client, args, in_path, out_path, semaphore)
            print('[%d/%d] %-75s sent %5d  shortcut %4d%s' % (n, len(files), rel, sent, cut,
                                                              ('  failed %d' % failed) if failed else ''))
            if failed:
                incomplete.append(rel)
    finally:
        await client.close()
    if incomplete:
        print('\n%d files still have error rows; rerun with --retry-errors:' % len(incomplete))
        for rel in incomplete:
            print('  ', rel)


if __name__ == '__main__':
    asyncio.run(main())
