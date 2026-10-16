#!/usr/bin/env python3
"""Run the compression benchmark through EXIT (doubleyyh/exit-gemma-2b on google/gemma-2b-it).

EXIT is an extractive compressor. It scores every sentence of the context for relevance
to the query and keeps the top k. The only option that changes what EXIT sees is the
context given with each sentence:

``--context-scope document``  Each sentence is scored against its own document,
    ``full_text = f"{title}\\n{text}" if title else text``. This is the setting used for
    the EXIT results in the paper. Outputs go to ``outputs/exit/gemma-2b-doc/``.

``--context-scope full``  Each sentence is scored against the whole context, built as
    ``" ".join(all_sentences)``: every sentence of every document joined by single spaces,
    so each title stays attached to the first sentence of its document and paragraph
    breaks are lost. Outputs go to ``outputs/exit/gemma-2b-fullctx/``.

Method
------
* Prompt:

      <start_of_turn>user
      Query:
      {query}
      Full context:
      {context}
      Sentence:
      {sentence}
      Is this sentence useful in answering the query? Answer only "Yes" or "No".<end_of_turn>
      <start_of_turn>model

* Score: softmax over ``logits[0, -1, [yes_id, no_id]]``, taking the first entry, i.e.
  P(Yes) normalized over the two candidate tokens only. ``yes_id`` and ``no_id`` are
  ``encode("Yes"/"No", add_special_tokens=False)[0]``.
* Sentence splitting: spaCy ``en_core_web_sm`` with every pipe disabled except ``senter``.
* Selection: the top k sentences by score, restored to their original order and joined
  with single spaces. One scoring pass serves all three budgets (0.5x / 1x / 2x).
* dtype: float16.

Merged weights. The LoRA adapter is merged into the base model once (``--stage merge``,
in float32 on CPU, saved as float16). The adapter is plain LoRA (no rsLoRA, no DoRA, no
``modules_to_save``, no bias), so ``W' = W + (alpha / r) B A`` is exactly what peft computes
at run time; the only numerical difference is one extra float16 rounding of the merged
weights, which is smaller than the float16 noise of the scoring pipeline itself.

Prefix KV reuse. The prompt up to ``Sentence:\\n`` is identical for all sentences that
share a context (one document, or one whole row for ``full``), so it is forwarded once
and the per-sentence suffixes are batched on top of its KV cache. ``--naive`` runs one
full prompt per sentence instead, and ``--stage verify`` compares the two.

Stages
------
* ``merge``   merge the adapter into the base model and write ``READY.json`` (CPU).
* ``split``   split every row into sentences, cached in
  ``outputs/_exit_sentences/<subset>.jsonl`` and shared by both scopes (CPU).
* ``verify``  compare prefix reuse against full prompts on a few rows (GPU).
* ``score``   score sentences and write the compressed outputs (GPU).

Usage
-----
    python src/compression/run_compression_exit.py --stage merge

    python src/compression/run_compression_exit.py --stage split --dry-run
    python src/compression/run_compression_exit.py --stage split

    python src/compression/run_compression_exit.py --stage verify
    python src/compression/run_compression_exit.py --stage score --context-scope document
    python src/compression/run_compression_exit.py --stage score --context-scope full
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('run_compression', HERE.parent / 'compression' / 'run_compression.py')
rc = importlib.util.module_from_spec(_spec)
sys.modules['run_compression'] = rc
_spec.loader.exec_module(rc)

DEFAULT_MODEL_DIR = HERE.parents[1] / 'models' / 'exit-gemma-2b'
BASE_REPO = 'google/gemma-2b-it'
ADAPTER_REPO = 'doubleyyh/exit-gemma-2b'
# The base repository also holds a GGUF export that the merge does not use.
BASE_PATTERNS = ['*.json', '*.safetensors', 'tokenizer.model']
OUTPUT_DIRS = {'document': 'exit/gemma-2b-doc', 'full': 'exit/gemma-2b-fullctx'}
SENTENCE_CACHE_DIR = '_exit_sentences'
# The EXIT prompt, split at "Sentence:\n": the head is shared within a group, the tail
# differs per sentence.
PROMPT_HEAD = ('<start_of_turn>user\n'
               'Query:\n%s\n'
               'Full context:\n%s\n'
               'Sentence:\n')
PROMPT_TAIL = ('%s\n'
               'Is this sentence useful in answering the query? '
               'Answer only "Yes" or "No".<end_of_turn>\n'
               '<start_of_turn>model\n')
DEFAULT_BATCH_SIZE = 64
# Token budget per batch of suffixes. Suffixes are short (about 40 tokens), so this only
# matters for unusually long sentences.
DEFAULT_MAX_BATCH_TOKENS = 8192


def prompt_of(query: str, context: str, sentence: str) -> str:
    """The complete EXIT prompt for one sentence."""
    return PROMPT_HEAD % (query, context) + PROMPT_TAIL % sentence


# --------------------------------------------------------------------------- #
# Subsets
# --------------------------------------------------------------------------- #

def subsets(args):
    """Return [(stem, input file, {budget: k})]."""
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
        entry[2][task.budget] = task.k
    return [(stem, path, ks) for stem, path, ks in plan.values()]


def cache_path(args, stem: str) -> Path:
    return args.output_root / SENTENCE_CACHE_DIR / ('%s.jsonl' % stem)


def output_path(args, stem: str, budget: str) -> Path:
    return (args.output_root / args.output_dir / 'without_thinking' / budget
            / ('%s.jsonl' % stem))


# --------------------------------------------------------------------------- #
# Stage split: EXIT sentence splitting (senter)
# --------------------------------------------------------------------------- #

def source_file(formatted: Path) -> Path:
    return formatted.with_name(formatted.stem + '.paragraphs.jsonl')


def load_paired(formatted: Path):
    rows = rc.load_jsonl(formatted)
    source = source_file(formatted)
    if not source.exists():
        raise SystemExit('Missing structured source file %s (it provides the paragraphs)' % source)
    srows = rc.load_jsonl(source)
    if len(rows) != len(srows):
        raise SystemExit('%s has %d rows, %s has %d rows'
                         % (formatted.name, len(rows), source.name, len(srows)))
    for i, (a, b) in enumerate(zip(rows, srows)):
        if a.get('id') != b.get('id'):
            raise SystemExit('Row %d id mismatch: %r vs %r' % (i, a.get('id'), b.get('id')))
        if not b.get('paragraphs'):
            raise SystemExit('%s row %d (%s) has no paragraphs' % (source.name, i, b.get('id')))
    return rows, srows


def load_splitter():
    """EXIT's spaCy configuration: every pipe disabled except senter."""
    import spacy

    nlp = spacy.load('en_core_web_sm',
                     disable=['tok2vec', 'tagger', 'parser', 'attribute_ruler',
                              'lemmatizer', 'ner'])
    nlp.enable_pipe('senter')
    return nlp


def row_sentences(nlp, srow):
    """Return (full_text of each document, sentences, owning document of each sentence).

    Sentences are in their original order, which is the order selection restores.
    """
    docs, sentences, owner = [], [], []
    for p in srow['paragraphs']:
        title = (p.get('title') or '').strip()
        text = p.get('paragraph_text', p.get('text', '')) or ''
        full_text = '%s\n%s' % (title, text) if title else text
        docs.append(full_text)
        for sent in nlp(full_text).sents:
            sent = sent.text.strip()
            if sent:
                sentences.append(sent)
                owner.append(len(docs) - 1)
    return docs, sentences, owner


def run_split(args):
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
        print('%d rows to split in total; spaCy en_core_web_sm, senter only' % total)
        return

    nlp = load_splitter()
    for stem, in_path, _ in work:
        cache = cache_path(args, stem)
        if cache.exists() and not args.overwrite:
            print('SKIP existing cache: %s' % cache)
            continue
        rows, srows = load_paired(in_path)
        if args.limit_per_task is not None:
            rows, srows = rows[:args.limit_per_task], srows[:args.limit_per_task]
        out, total = [], 0
        for row, srow in zip(tqdm(rows, desc=stem), srows):
            docs, sentences, owner = row_sentences(nlp, srow)
            if not sentences:
                raise SystemExit('%s row %s yields no sentences' % (stem, row.get('id')))
            out.append({'id': row.get('id'), 'query': (row.get('question') or '').strip(),
                        'docs': docs, 'sentences': sentences, 'owner': owner})
            total += len(sentences)
        rc.write_jsonl(cache, out)
        print('DONE: %s  %d rows / %d sentences (%.1f sentences/row)'
              % (cache, len(out), total, total / len(out)))


# --------------------------------------------------------------------------- #
# Stage merge
# --------------------------------------------------------------------------- #

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hub_snapshot(repo_id: str, allow_patterns=None) -> Path:
    """Download (or reuse the cached) snapshot of a Hugging Face repository."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, allow_patterns=allow_patterns))


def run_merge(args):
    """Merge the adapter into the base model and write READY.json. Only this stage needs peft."""
    adapter = args.adapter or hub_snapshot(ADAPTER_REPO)
    base = (args.base or hub_snapshot(BASE_REPO, BASE_PATTERNS)).resolve()

    config = json.loads((adapter / 'adapter_config.json').read_text())
    if config['base_model_name_or_path'] != 'google/gemma-2b-it':
        raise SystemExit('The adapter declares base model %s, not google/gemma-2b-it'
                         % config['base_model_name_or_path'])
    if not (base / 'config.json').is_file():
        raise SystemExit('Incomplete base model snapshot (no config.json): %s' % base)

    adapter_hash = sha256(adapter / 'adapter_model.safetensors')
    ready = args.model_dir / 'READY.json'
    if ready.is_file():
        saved = json.loads(ready.read_text())
        if saved['adapter_sha256'] == adapter_hash:
            print('Reusing verified merged model: %s' % args.model_dir)
            return
        raise SystemExit('READY.json in %s belongs to a different adapter; use a new directory'
                         % args.model_dir)
    if args.model_dir.exists() and any(args.model_dir.iterdir()):
        raise SystemExit('%s is not empty and has no READY.json; use a new directory'
                         % args.model_dir)

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    print('Base:    %s' % base)
    print('Adapter: %s  (r=%d, alpha=%d -> scaling=%.3f)'
          % (adapter, config['r'], config['lora_alpha'], config['lora_alpha'] / config['r']))
    print('Loading the base model on CPU (float32, about 10 GB)...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, dtype=torch.float32, device_map='cpu')

    print('Attaching the adapter...', flush=True)
    peft_model = PeftModel.from_pretrained(model, adapter, local_files_only=True)

    # peft does not fail when target_modules do not match; it creates fewer LoRA layers
    # and the merge silently degrades to the base model. Check every saved tensor.
    saved = load_file(adapter / 'adapter_model.safetensors')
    live = dict(peft_model.named_parameters())
    missing = []
    for key in saved:
        candidates = [key, key.replace('.lora_A.weight', '.lora_A.default.weight')
                             .replace('.lora_B.weight', '.lora_B.default.weight')]
        if not any(c in live for c in candidates):
            missing.append(key)
    print('Adapter tensors: %d, matched in the model: %d' % (len(saved), len(saved) - len(missing)))
    if missing:
        raise SystemExit('%d LoRA tensors were not attached (first few: %s). The peft version '
                         'or target_modules do not match; the merge would equal the base model.'
                         % (len(missing), missing[:3]))

    print('Merging (safe_merge, float32)...', flush=True)
    merged = peft_model.merge_and_unload(safe_merge=True)

    print('Casting to float16 and saving to %s...' % args.model_dir, flush=True)
    merged = merged.to(torch.float16)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.model_dir, safe_serialization=True, max_shard_size='4GB')
    # EXIT uses the base model's tokenizer, not the one in the adapter repository.
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(args.model_dir)

    weights = {p.name: sha256(p) for p in sorted(args.model_dir.glob('*.safetensors'))}
    manifest = {
        'merged_model': 'doubleyyh/exit-gemma-2b merged into google/gemma-2b-it',
        'base_path': str(base), 'adapter_path': str(adapter),
        'adapter_sha256': adapter_hash,
        'lora_r': config['r'], 'lora_alpha': config['lora_alpha'],
        'target_modules': sorted(config['target_modules']),
        'merge_dtype': 'float32', 'saved_dtype': 'float16',
        'adapter_tensors': len(saved),
        'weights_sha256': weights,
        'merged_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    ready.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('Done in %.1f s; %d weight shards' % (time.perf_counter() - started, len(weights)))


# --------------------------------------------------------------------------- #
# Stage score
# --------------------------------------------------------------------------- #

def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available (torch %s). --stage score and --stage verify '
                         'need a GPU.' % torch.__version__)
    if torch.cuda.device_count() != 1:
        raise SystemExit('Found %d GPUs. Select one with CUDA_VISIBLE_DEVICES.'
                         % torch.cuda.device_count())
    ready = args.model_dir / 'READY.json'
    if not ready.is_file():
        raise SystemExit('No READY.json in %s; run --stage merge first' % args.model_dir)
    device = torch.device('cuda')
    # The tokenizer is read from the base model, not from the merged directory. EXIT uses
    # the base tokenizer, and the copy written by transformers 5 stores
    # ``extra_special_tokens`` as a list, which transformers 4.51 cannot read.
    tokenizer_dir = args.tokenizer or Path(json.loads(ready.read_text())['base_path'])
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=dtype).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=dtype).to(device)
    model.eval()
    yes_id = tok.encode('Yes', add_special_tokens=False)[0]
    no_id = tok.encode('No', add_special_tokens=False)[0]
    print('Loaded %s on %s (%s, yes_id=%d, no_id=%d)'
          % (args.model_dir.name, torch.cuda.get_device_name(0), args.dtype, yes_id, no_id))
    return tok, model, device, yes_id, no_id


def score_naive(tok, model, device, query, contexts, sentences, yes_id, no_id, args):
    """One complete prompt per sentence, one forward pass each."""
    import torch

    scores = []
    with torch.no_grad():
        for context, sentence in zip(contexts, sentences):
            inputs = tok(prompt_of(query, context, sentence), return_tensors='pt').to(device)
            logits = model(**inputs).logits[0, -1, [yes_id, no_id]]
            scores.append(torch.softmax(logits, dim=0)[0].item())
    return scores


def score_grouped(tok, model, device, query, groups, yes_id, no_id, args):
    """Prefix KV reuse. ``groups`` is [(context, [(sentence index, sentence), ...]), ...].

    Each group forwards ``PROMPT_HEAD % (query, context)`` once and batches the sentence
    suffixes on top of its KV cache. Padding follows the real tokens and attention is
    causal, and the logits are read at each row's last real position, so padding does
    not affect the scores.
    """
    import torch
    from transformers import DynamicCache

    scores = {}
    with torch.no_grad():
        for context, items in groups:
            head_ids = tok(PROMPT_HEAD % (query, context), return_tensors='pt').input_ids.to(device)
            prefix_len = head_ids.shape[1]
            prefix = model(head_ids, use_cache=True).past_key_values
            layers = [(k, v) for k, v in zip(prefix.key_cache, prefix.value_cache)] \
                if hasattr(prefix, 'key_cache') else list(prefix)

            tails = [tok(PROMPT_TAIL % sentence, add_special_tokens=False).input_ids
                     for _, sentence in items]
            order = sorted(range(len(items)), key=lambda i: len(tails[i]))
            batch, longest = [], 0
            batches = []
            for i in order:
                nxt = max(longest, len(tails[i]))
                if batch and (len(batch) + 1 > args.batch_size
                              or nxt * (len(batch) + 1) > args.max_batch_tokens):
                    batches.append(batch)
                    batch, longest = [i], len(tails[i])
                else:
                    batch.append(i)
                    longest = nxt
            if batch:
                batches.append(batch)

            for chunk in batches:
                size = len(chunk)
                width = max(len(tails[i]) for i in chunk)
                ids = torch.full((size, width), tok.pad_token_id or 0,
                                 dtype=torch.long, device=device)
                mask = torch.zeros((size, prefix_len + width), dtype=torch.long, device=device)
                mask[:, :prefix_len] = 1
                last = []
                for row, i in enumerate(chunk):
                    tail = tails[i]
                    ids[row, :len(tail)] = torch.tensor(tail, device=device)
                    mask[row, prefix_len:prefix_len + len(tail)] = 1
                    last.append(len(tail) - 1)
                positions = (prefix_len + torch.arange(width, device=device)).unsqueeze(0).expand(size, -1)
                cache = DynamicCache.from_legacy_cache(
                    tuple((k.expand(size, -1, -1, -1).contiguous(),
                           v.expand(size, -1, -1, -1).contiguous()) for k, v in layers))
                out = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                            past_key_values=cache, use_cache=False)
                picked = out.logits[torch.arange(size, device=device),
                                    torch.tensor(last, device=device)][:, [yes_id, no_id]]
                probs = torch.softmax(picked, dim=1)[:, 0].tolist()
                for i, p in zip(chunk, probs):
                    scores[items[i][0]] = p
    return [scores[i] for i in sorted(scores)]


def contexts_for(entry, scope: str):
    """The context given with each sentence; the only difference between the two scopes.

    document: the full_text of the sentence's own document.
    full:     " ".join(all sentences of the row).
    """
    if scope == 'document':
        return [entry['docs'][o] for o in entry['owner']]
    joined = ' '.join(entry['sentences'])
    return [joined] * len(entry['sentences'])


def group_by_context(entry, scope: str):
    """Group sentences by context, keeping the original order within each group."""
    if scope == 'full':
        joined = ' '.join(entry['sentences'])
        return [(joined, list(enumerate(entry['sentences'])))]
    groups = []
    for doc_index, doc in enumerate(entry['docs']):
        items = [(i, s) for i, (s, o) in enumerate(zip(entry['sentences'], entry['owner']))
                 if o == doc_index]
        if items:
            groups.append((doc, items))
    return groups


def compose(sentences, scores, k):
    """Take the top k sentences by score, restore their original order, join with spaces."""
    order = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    chosen = sorted(order[:min(k, len(sentences))])
    return ' '.join(sentences[i] for i in chosen), len(chosen)


def run_score(args):
    work = subsets(args)
    plans = []
    for stem, in_path, ks in work:
        plans.append((stem, in_path, ks, cache_path(args, stem),
                      {b: output_path(args, stem, b) for b in rc.BUDGETS}))
    missing = [str(c) for _, _, _, c, _ in plans if not c.exists()]
    if missing:
        raise SystemExit('Missing sentence caches; run --stage split first:\n  '
                         + '\n  '.join(missing))

    print('context-scope=%s -> %s' % (args.context_scope, args.output_dir))
    if args.dry_run:
        rows_total = sent_total = prefix_total = tail_total = 0
        for stem, in_path, ks, cache, outs in plans:
            rows = rc.load_jsonl(cache)
            sents = sum(len(r['sentences']) for r in rows)
            groups = sum(len(group_by_context(r, args.context_scope)) for r in rows)
            rows_total += len(rows)
            sent_total += sents
            prefix_total += groups
            tail_total += sents
            done = [b for b in rc.BUDGETS if outs[b].exists()]
            print('  %-34s %5d rows %7d sentences  prefix groups %6d (%.1f sentences/group)  k=%s%s'
                  % (stem, len(rows), sents, groups, sents / groups,
                     '/'.join(str(ks[b]) for b in rc.BUDGETS),
                     ('  exists: ' + ','.join(done)) if done else ''))
        print('Total: %d rows / %d sentences / %d prefix groups; %d files / %d rows to write'
              % (rows_total, sent_total, prefix_total,
                 len(plans) * len(rc.BUDGETS), rows_total * len(rc.BUDGETS)))
        tok_sample = rc.load_jsonl(plans[0][3])[0]
        ctx = contexts_for(tok_sample, args.context_scope)[0]
        print('\nExample prompt (%s, first sentence of the first row, context of %d chars):\n%s'
              % (plans[0][0], len(ctx),
                 prompt_of(tok_sample['query'], ctx[:400] + (' …' if len(ctx) > 400 else ''),
                           tok_sample['sentences'][0])))
        return

    tok, model, device, yes_id, no_id = load_model(args)
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
            raise SystemExit('%s: %d input rows but %d cached rows; the cache is stale, '
                             're-split with --overwrite' % (stem, len(rows), len(cached)))
        for row, entry in zip(rows, cached):
            if row.get('id') != entry.get('id'):
                raise SystemExit('%s: cache and input ids are misaligned (%r vs %r)'
                                 % (stem, row.get('id'), entry.get('id')))

        results = {b: [] for b in pending}
        for row, entry in zip(tqdm(rows, desc='%s %s' % (stem, ','.join(pending))), cached):
            sentences = entry['sentences']
            if args.naive:
                scores = score_naive(tok, model, device, entry['query'],
                                     contexts_for(entry, args.context_scope),
                                     sentences, yes_id, no_id, args)
            else:
                scores = score_grouped(tok, model, device, entry['query'],
                                       group_by_context(entry, args.context_scope),
                                       yes_id, no_id, args)
            for budget in pending:
                text, selected = compose(sentences, scores, ks[budget])
                out = dict(row)
                out['compressed_context'] = text if text else rc.ERROR_COMPRESSED_CONTEXT
                out['exit_sentences_total'] = len(sentences)
                out['exit_selected'] = selected
                out['exit_k'] = ks[budget]
                out['exit_context_scope'] = args.context_scope
                results[budget].append(out)

        for budget in pending:
            rc.write_jsonl(outs[budget], results[budget])
            failed = sum(1 for r in results[budget] if rc.is_failed(r))
            print('DONE: %s%s' % (outs[budget], '  (%d failed)' % failed if failed else ''))


# --------------------------------------------------------------------------- #
# Stage verify: prefix reuse vs full prompts
# --------------------------------------------------------------------------- #

def run_verify(args):
    import torch

    tok, model, device, yes_id, no_id = load_model(args)
    work = subsets(args)
    worst = {}
    for scope in ('document', 'full'):
        gaps, token_mismatch, checked = [], 0, 0
        for stem, in_path, ks in work[:args.verify_subsets]:
            cache = cache_path(args, stem)
            if not cache.exists():
                raise SystemExit('Missing sentence cache %s' % cache)
            for entry in rc.load_jsonl(cache)[:args.verify_rows]:
                # Both paths must see the same context, so trim the row first and derive the
                # contexts from the trimmed row: trimming changes the full-scope context.
                sentences = entry['sentences'][:args.verify_sentences]
                trimmed = {'query': entry['query'], 'docs': entry['docs'],
                           'sentences': sentences,
                           'owner': entry['owner'][:len(sentences)]}
                contexts = contexts_for(trimmed, scope)
                # Token boundary: head ids + tail ids must equal the ids of the full prompt.
                for context, sentence in zip(contexts, sentences):
                    head = tok(PROMPT_HEAD % (entry['query'], context)).input_ids
                    tail = tok(PROMPT_TAIL % sentence, add_special_tokens=False).input_ids
                    full = tok(prompt_of(entry['query'], context, sentence)).input_ids
                    token_mismatch += head + tail != full
                    checked += 1
                naive = score_naive(tok, model, device, entry['query'], contexts,
                                    sentences, yes_id, no_id, args)
                fast = score_grouped(tok, model, device, entry['query'],
                                     group_by_context(trimmed, scope), yes_id, no_id, args)
                gaps.extend(abs(a - b) for a, b in zip(naive, fast))
        worst[scope] = (max(gaps), sum(gaps) / len(gaps), len(gaps), token_mismatch, checked)
        print('%-9s %4d sentences compared: |dP(Yes)| max %.2e, mean %.2e; '
              'token boundary mismatches %d/%d'
              % (scope, len(gaps), max(gaps), sum(gaps) / len(gaps), token_mismatch, checked))
    if any(v[0] > args.verify_tolerance for v in worst.values()):
        raise SystemExit('Prefix reuse differs from full prompts by more than %g; run '
                         '--stage score with --naive' % args.verify_tolerance)
    print('Prefix reuse matches full prompts within tolerance %g.' % args.verify_tolerance)


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='Context compression with EXIT (gemma-2b + LoRA).')
    p.add_argument('--stage', required=True,
                   choices=['merge', 'split', 'score', 'verify'],
                   help='merge and split run on CPU (peft / spaCy); score and verify need a GPU')
    p.add_argument('--context-scope', choices=['document', 'full'], default='document',
                   help='document: each sentence is scored against its own document (paper '
                        'setting); full: each sentence is scored against the whole context')
    p.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                   help='merged weights: written by --stage merge, read by score/verify '
                        '(default: %(default)s)')
    p.add_argument('--adapter', type=Path,
                   help='--stage merge only; local adapter directory '
                        '(default: Hugging Face snapshot of %s)' % ADAPTER_REPO)
    p.add_argument('--base', type=Path,
                   help='--stage merge only; local base model directory '
                        '(default: Hugging Face snapshot of %s)' % BASE_REPO)
    p.add_argument('--threads', type=int, default=16,
                   help='CPU threads for --stage merge, default %(default)s')
    p.add_argument('--tokenizer', type=Path,
                   help='default: the base model path recorded in READY.json')
    p.add_argument('--data-root', type=Path, default=rc.DEFAULT_DATA_ROOT)
    p.add_argument('--output-root', type=Path, default=rc.DEFAULT_OUTPUT_ROOT)
    p.add_argument('--shuffle-seed', type=int, default=rc.DEFAULT_SHUFFLE_SEED)
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                   help='maximum sentence suffixes per batch within a group, default %(default)s')
    p.add_argument('--max-batch-tokens', type=int, default=DEFAULT_MAX_BATCH_TOKENS)
    p.add_argument('--dtype', default='float16',
                   choices=['float16', 'bfloat16', 'float32'],
                   help='EXIT uses float16; gemma is less stable in float16 on long contexts, '
                        'see --stage verify')
    p.add_argument('--naive', action='store_true',
                   help='one full-prompt forward pass per sentence (slow; avoid for --context-scope full)')
    p.add_argument('--verify-rows', type=int, default=2)
    p.add_argument('--verify-subsets', type=int, default=3)
    p.add_argument('--verify-sentences', type=int, default=24)
    p.add_argument('--verify-tolerance', type=float, default=5e-3)
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
    args.output_dir = OUTPUT_DIRS[args.context_scope]
    args.budget = 'all'
    args.family = 'exit'
    args.enable_thinking = False
    return args


def main():
    args = parse_args()
    if args.stage == 'merge':
        run_merge(args)
    elif args.stage == 'split':
        run_split(args)
    elif args.stage == 'score':
        run_score(args)
    else:
        run_verify(args)


if __name__ == '__main__':
    main()
