#!/usr/bin/env python3
r"""Run ICL context compression over the whole benchmark at 0.5x / 1x / 2x.

One script covers every locally served model family (Qwen3, Llama-3.1/3.2,
Gemma-4) through an OpenAI-compatible endpoint such as vLLM.

    27 tasks = 3 settings x 9 subsets x 3 budgets

      parallel      two_hop / three_hop / four_hop          x 0.5x 1x 2x
      conditional   two_hop / three_hop / four_hop          x 0.5x 1x 2x
      imbalanced    science three_perspectives              x 0.5x 1x 2x
                    science four_perspectives               x 0.5x 1x 2x
                    policy  four_perspectives               x 0.5x 1x 2x

Each task reads one input file under ``data/`` (one ``context`` string per
row), few-shot prompts the model with that setting's ICL demonstrations, and
writes the input row back out with one added key, ``compressed_context``.

Sentence budgets per subset (``k``); 1x is the default budget in the paper:

    subset                        0.5x   1x   2x
    parallel/conditional two_hop   2    4    8
    parallel/conditional three_hop 3    6   12
    parallel/conditional four_hop  4    8   16
    science three_perspectives     4    7   13
    science/policy four_persp.     5    9   17

The ICL assistant turns come from ``data/icl/<setting>.jsonl``, which stores
one reference compression per budget. The parallel and conditional ICL files
mix hops, so the hop of each example is recovered from its logical-chain length
and the matching budget field is selected; the resulting hop distribution is
asserted against ``MIXED_HOP_ICL_COUNTS``. Shot counts per setting are fixed:
parallel 6, conditional 9, science 5, policy 3. The demonstrations are shuffled
with a fixed seed (``--shuffle-seed``, default 42).

Every model family sees exactly the same prompts, so results from different
families are directly comparable.

Context length
--------------
The ICL prefix is sent with every request, so the served context window
matters. Exact counts over every row of the benchmark, Qwen3 tokenizer:

    setting       shots   ICL prefix (0.5x/1x/2x)   longest prompt
    policy            3     11651 / 11960 / 12585          17227
    parallel          6     13199 / 13767 / 14906          20492
    science           5     16459 / 16651 / 17027          22138
    conditional       9     15573 / 16178 / 17459          22497

The full benchmark therefore needs ``22497 + max_tokens``; with the default
``--max-tokens 1024`` that is 23521, which fits in a 32768 window. The default
of 1024 output tokens leaves headroom over the longest designed target (640
tokens, parallel four_hop at 2x) and cuts off runaway generations early. Served
context windows:

    Qwen3 4B/8B/14B/32B    40960 (config max_position_embeddings)
    Llama-3.1 / 3.2       131072
    Gemma-4                set by the server's --max-model-len

Qwen3 was pretrained at 32768; its shipped configs extend this to 40960, which
vLLM serves unless ``--max-model-len`` says otherwise.

Preflight tokenizes each task's longest rows through vLLM's ``/tokenize`` and
aborts with the required ``--max-model-len`` before any generation starts, so a
too-small window is caught in seconds instead of after a full task of error
sentinels.

Serving
-------
``bash src/compression/serve_vllm.sh <alias>`` starts the server for a registry
alias. It fixes the two settings that affect results or how they are read:
``--served-model-name`` (the id this script sends) and
``--enable-prefix-caching`` (the FLOPs accounting in ``src/computation/``
charges each setting one prefix prefill, which holds only with the cache on).
``--tensor-parallel-size``, ``--gpu-memory-utilization`` and
``--max-model-len`` depend on the hardware and are passed through.

Usage
-----
    # list the known model aliases
    python src/compression/run_compression.py --list-models

    # start the server, then run everything at all three budgets
    bash src/compression/serve_vllm.sh qwen3-8b
    python src/compression/run_compression.py --model qwen3-8b

    # a single setting / budget, or a short smoke run
    python src/compression/run_compression.py --model qwen3-8b --setting conditional --budget 1x
    python src/compression/run_compression.py --model qwen3-8b --limit-per-task 4 --dry-run

    # a server on another port
    python src/compression/run_compression.py --model gemma4-e4b --base-url http://localhost:8001/v1

    # a model that is not in the registry
    python src/compression/run_compression.py --served-model my-model --family qwen3 \
        --base-url http://localhost:8000/v1 --output-dir custom/my-model

Existing outputs are skipped unless ``--overwrite`` is given. ``--retry-errors``
keeps a finished file and re-requests only the rows that failed (empty output or
the ``ERROR_COMPRESSED_CONTEXT`` sentinel).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"

# Written into compressed_context when every retry of a request failed.
# --retry-errors re-requests exactly these rows (and rows with empty output).
ERROR_COMPRESSED_CONTEXT = "Null character resulting from an error"

DEFAULT_SHUFFLE_SEED = 42
DEFAULT_MAX_CONCURRENCY = 64
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT = 300.0
# The largest reference compression in the ICL data is 640 tokens (parallel four_hop
# at 2x). 1024 leaves headroom over it, keeps vLLM's per-request reservation small,
# which reduces preemption, and truncates runaway generations early.
DEFAULT_MAX_TOKENS = 1024
TOKENIZE_PROBE_TIMEOUT = 30.0     # per-probe timeout in preflight_context

BUDGETS = ("0.5x", "1x", "2x")
HOPS = ("two_hop", "three_hop", "four_hop")
PERSPECTIVES = ("three_perspectives", "four_perspectives")


# --------------------------------------------------------------------------- #
# Budgets and ICL wiring
# --------------------------------------------------------------------------- #

PARALLEL_K = {
    "two_hop": {"0.5x": 2, "1x": 4, "2x": 8},
    "three_hop": {"0.5x": 3, "1x": 6, "2x": 12},
    "four_hop": {"0.5x": 4, "1x": 8, "2x": 16},
}
PARALLEL_ICL_KEYS = {
    "two_hop": {"0.5x": "70_tokens_compressed_context", "1x": "140_tokens_compressed_context", "2x": "280_tokens_compressed_context"},
    "three_hop": {"0.5x": "110_tokens_compressed_context", "1x": "220_tokens_compressed_context", "2x": "440_tokens_compressed_context"},
    "four_hop": {"0.5x": "160_tokens_compressed_context", "1x": "320_tokens_compressed_context", "2x": "640_tokens_compressed_context"},
}

CONDITIONAL_K = {
    "two_hop": {"0.5x": 2, "1x": 4, "2x": 8},
    "three_hop": {"0.5x": 3, "1x": 6, "2x": 12},
    "four_hop": {"0.5x": 4, "1x": 8, "2x": 16},
}
CONDITIONAL_ICL_KEYS = {
    "two_hop": {"0.5x": "40_tokens_compressed_result", "1x": "80_tokens_compressed_result", "2x": "160_tokens_compressed_result"},
    "three_hop": {"0.5x": "62_tokens_compressed_result", "1x": "124_tokens_compressed_result", "2x": "248_tokens_compressed_result"},
    "four_hop": {"0.5x": "87_tokens_compressed_result", "1x": "174_tokens_compressed_result", "2x": "348_tokens_compressed_result"},
}

# Hop distribution inside the mixed-hop ICL files, asserted on every load.
MIXED_HOP_ICL_COUNTS = {
    "parallel": {"two_hop": 1, "three_hop": 2, "four_hop": 3},
    "conditional": {"two_hop": 2, "three_hop": 3, "four_hop": 4},
}

SCIENCE_K = {
    "three_perspectives": {"0.5x": 4, "1x": 7, "2x": 13},
    "four_perspectives": {"0.5x": 5, "1x": 9, "2x": 17},
}
POLICY_K = {"four_perspectives": {"0.5x": 5, "1x": 9, "2x": 17}}

# Total shots per setting, asserted on every load.
EXPECTED_SHOTS = {"parallel": 6, "conditional": 9, "science": 5, "policy": 3}

# Paths inside the data root.
PARALLEL_INPUT = "parallel/{hop}.jsonl"
PARALLEL_ICL = "icl/parallel.jsonl"
CONDITIONAL_INPUT = "conditional/{hop}.jsonl"
CONDITIONAL_ICL = "icl/conditional.jsonl"
SCIENCE_INPUT = "science/{perspective}.jsonl"
SCIENCE_ICL = "icl/science.jsonl"
POLICY_INPUT = "policy/four_perspectives.jsonl"
POLICY_ICL = "icl/policy.jsonl"


# --------------------------------------------------------------------------- #
# Model families
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FamilySpec:
    """What differs between served model families at request/response level."""

    # vLLM forwards chat_template_kwargs into the chat template. Qwen3 and
    # Gemma-4 templates read enable_thinking; the Llama-3 templates do not, so
    # nothing is sent for them.
    supports_thinking: bool
    # Gemma-4 can emit channel sentinels around its final answer.
    gemma_channels: bool = False


FAMILIES = {
    "qwen3": FamilySpec(supports_thinking=True),
    "llama3": FamilySpec(supports_thinking=False),
    "gemma4": FamilySpec(supports_thinking=True, gemma_channels=True),
}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    served_model: str   # the id the server lists at its models endpoint
    output_dir: str     # <output-root>/<output_dir>/<thinking tag>/<budget>/
    hub_id: str         # Hugging Face repository that serve_vllm.sh loads
    base_url: str = "http://localhost:8000/v1"   # override with --base-url


MODELS = {
    "qwen3-4b": ModelSpec("qwen3", "Qwen3-4B", "qwen3/4B", "Qwen/Qwen3-4B"),
    "qwen3-8b": ModelSpec("qwen3", "Qwen3-8B", "qwen3/8B", "Qwen/Qwen3-8B"),
    "qwen3-14b": ModelSpec("qwen3", "Qwen3-14B", "qwen3/14B", "Qwen/Qwen3-14B"),
    "qwen3-32b": ModelSpec("qwen3", "Qwen3-32B", "qwen3/32B", "Qwen/Qwen3-32B"),
    "llama3.2-1b": ModelSpec("llama3", "Llama-3.2-1B-Instruct", "llama3/1B", "meta-llama/Llama-3.2-1B-Instruct"),
    "llama3.2-3b": ModelSpec("llama3", "Llama-3.2-3B-Instruct", "llama3/3B", "meta-llama/Llama-3.2-3B-Instruct"),
    "llama3.1-8b": ModelSpec("llama3", "Llama-3.1-8B-Instruct", "llama3/8B", "meta-llama/Llama-3.1-8B-Instruct"),
    "gemma4-e2b": ModelSpec("gemma4", "Gemma-4-E2B-it", "gemma4/E2B", "google/gemma-4-E2B-it"),
    "gemma4-e4b": ModelSpec("gemma4", "Gemma-4-E4B-it", "gemma4/E4B", "google/gemma-4-E4B-it"),
    "gemma4-12b": ModelSpec("gemma4", "google/gemma-4-12B-it", "gemma4/12B", "google/gemma-4-12B-it"),
    "gemma4-31b": ModelSpec("gemma4", "google/gemma-4-31B-it", "gemma4/31B", "google/gemma-4-31B-it"),
}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def parallel_system_prompt() -> str:
    return (
        "Write a high-quality summary of the provided documents with respect to the question.\n"
        "The question may contain potentially ambiguous entities, and within the documents some "
        "entities mentioned in the question may refer to multiple different referents.\n"
        "Preserve all such possible interpretations and supporting evidence in the summary.\n"
        "Include all relevant evidence for different possible referents, preserve conflicting "
        "evidence when present, and prioritize and front-load the most critical points while "
        "remaining concise."
    )


def conditional_system_prompt() -> str:
    return (
        "Write a high-quality summary of the provided documents with respect to the question.\n"
        "The question may have different correct answers under different conditions.\n"
        "Preserve the condition-bearing evidence needed for an answer to be correct, especially "
        "when a result holds only under a specific condition.\n"
        "Prioritize and front-load the most critical evidence needed to answer the question, and "
        "do not omit the condition if the answer depends on it."
    )


def diverse_science_system_prompt() -> str:
    return (
        "Write a concise, document-grounded summary of the provided documents with respect to the question.\n"
        "The summary should stay focused on the question, preserve the answer categories supported by the documents, "
        "and maximize coverage of distinct perspectives when the documents support multiple answers.\n"
        "Do not add facts that are not supported by the documents.\n"
        "Prefer short, direct sentences that front-load the most important evidence."
    )


def diverse_policy_system_prompt() -> str:
    return (
        "Write a concise, document-grounded summary of the provided documents with respect to the question.\n"
        "The summary should stay focused on the question, preserve the answer categories supported by the documents, "
        "and maximize coverage of distinct perspectives when the documents support multiple answers.\n"
        "For policy questions, each answer category must clearly identify the applicable country, region, legal system, "
        "or jurisdiction.\n"
        "Do not add facts that are not supported by the documents.\n"
        "Prefer short, direct sentences that front-load the most important evidence."
    )


def system_prompt(prompt_family: str) -> str:
    if prompt_family == "parallel":
        return parallel_system_prompt()
    if prompt_family == "conditional":
        return conditional_system_prompt()
    if prompt_family == "diverse_science":
        return diverse_science_system_prompt()
    if prompt_family == "diverse_policy":
        return diverse_policy_system_prompt()
    raise ValueError(f"Unknown prompt family: {prompt_family}")


def user_prompt(question: str, context: str, k: int, prompt_family: str) -> str:
    if prompt_family == "parallel":
        instruction = (
            "must include all relevant evidence for different possible referents; "
            "prioritize and front-load the most critical points"
        )
    elif prompt_family == "conditional":
        instruction = (
            "must preserve condition-bearing evidence; "
            "prioritize and front-load the most critical points"
        )
    elif prompt_family == "diverse_policy":
        instruction = (
            "must stay focused on the question; "
            "must remain grounded in the documents; "
            "must maximize coverage of distinct supported perspectives; "
            "must identify the applicable jurisdiction for each policy answer"
        )
    elif prompt_family == "diverse_science":
        instruction = (
            "must stay focused on the question; "
            "must remain grounded in the documents; "
            "must maximize coverage of distinct supported perspectives"
        )
    else:
        raise ValueError(f"Unknown prompt family: {prompt_family}")

    return (
        f"### This is the question: {question}\n"
        f"### These are the documents:\n{context}\n"
        f"### This is the summary ({instruction}; total length must not exceed {k} sentences):"
    )


def build_messages(entry, k: int, prompt_family: str, icl_examples):
    """system turn, then the ICL user/assistant pairs, then the real query."""
    messages = [{"role": "system", "content": system_prompt(prompt_family)}]
    for example in icl_examples:
        messages.append(
            {
                "role": "user",
                "content": user_prompt(
                    example["question"], example["context"], example["k"], prompt_family
                ),
            }
        )
        messages.append({"role": "assistant", "content": example["assistant"]})
    messages.append(
        {
            "role": "user",
            "content": user_prompt(
                entry.get("question", ""), entry.get("context", ""), k, prompt_family
            ),
        }
    )
    return messages


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {error}") from error
    return rows


def write_jsonl(path: Path, rows):
    """Write through a temporary file so an interrupted run cannot truncate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def str2bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# --------------------------------------------------------------------------- #
# ICL loading
# --------------------------------------------------------------------------- #

def load_mixed_hop_icl(setting: str, data_root: Path, budget: str, shuffle_seed: int | None):
    """parallel / conditional: one ICL file mixing hops.

    The hop of each example is recovered from its logical-chain length, which
    then selects both the reference compression field and the sentence budget.
    """
    if setting == "parallel":
        path = data_root / PARALLEL_ICL
        keys, limits, chain_field = PARALLEL_ICL_KEYS, PARALLEL_K, "logical_chain_1"
    elif setting == "conditional":
        path = data_root / CONDITIONAL_ICL
        keys, limits, chain_field = CONDITIONAL_ICL_KEYS, CONDITIONAL_K, "logical_chain"
    else:
        raise ValueError(f"Unknown mixed-hop ICL setting: {setting}")

    hop_counts = {hop: 0 for hop in HOPS}
    examples = []
    for row_index, row in enumerate(load_jsonl(path), start=1):
        hop = {2: "two_hop", 3: "three_hop", 4: "four_hop"}.get(len(row[chain_field]))
        if hop is None:
            raise ValueError(f"Unexpected chain length in {path} row {row_index}")
        hop_counts[hop] += 1
        assistant_key = keys[hop][budget]
        if assistant_key not in row:
            raise KeyError(f"Missing key '{assistant_key}' in {path} row {row_index}")
        examples.append(
            {
                "question": row["question"],
                "context": row["context"],
                "assistant": row[assistant_key],
                "k": limits[hop][budget],
            }
        )

    expected = MIXED_HOP_ICL_COUNTS[setting]
    if hop_counts != expected:
        raise ValueError(f"Expected {setting} ICL hop counts {expected}, got {hop_counts} in {path}")
    return shuffled(examples, shuffle_seed)


def load_perspective_icl(domain: str, data_root: Path, budget: str, shuffle_seed: int | None):
    """science / policy: one ICL file, budget picked per perspective count."""
    if domain == "science":
        path, limits = data_root / SCIENCE_ICL, SCIENCE_K
        allowed = {3: "three_perspectives", 4: "four_perspectives"}
    elif domain == "policy":
        path, limits = data_root / POLICY_ICL, POLICY_K
        allowed = {4: "four_perspectives"}
    else:
        raise ValueError(f"Unknown perspective ICL domain: {domain}")

    examples = []
    for row_index, row in enumerate(load_jsonl(path), start=1):
        count = len(row.get("perspectives", []))
        perspective = allowed.get(count)
        if perspective is None:
            raise ValueError(
                f"Expected {sorted(allowed)} perspectives in {path} row {row_index}, got {count}"
            )
        k = limits[perspective][budget]
        assistant_key = f"{k}_sentences_compressed_result"
        if assistant_key not in row:
            raise KeyError(f"Missing key '{assistant_key}' in {path} row {row_index}")
        examples.append(
            {
                "question": row["question"],
                "context": row["context"],
                "assistant": row[assistant_key],
                "k": k,
            }
        )
    return shuffled(examples, shuffle_seed)


def shuffled(examples, shuffle_seed: int | None):
    shuffler = random.Random(shuffle_seed) if shuffle_seed is not None else random
    shuffler.shuffle(examples)
    return examples


def load_icl_examples(task: "TaskSpec", data_root: Path, shuffle_seed: int | None):
    if task.icl_family in {"parallel", "conditional"}:
        examples = load_mixed_hop_icl(task.icl_family, data_root, task.budget, shuffle_seed)
    elif task.icl_family in {"science", "policy"}:
        examples = load_perspective_icl(task.icl_family, data_root, task.budget, shuffle_seed)
    else:
        raise ValueError(f"Unknown ICL family: {task.icl_family}")

    expected = EXPECTED_SHOTS[task.icl_family]
    if len(examples) != expected:
        raise ValueError(
            f"Expected {expected} {task.icl_family} ICL shots, got {len(examples)}"
        )
    return examples


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TaskSpec:
    name: str
    setting: str            # parallel | conditional | imbalanced
    input_file: Path
    output_file: Path
    k: int
    prompt_family: str      # selects the prompt text
    icl_family: str         # selects the ICL file and budget table
    budget: str
    hop: str | None = None
    domain: str | None = None
    perspective: str | None = None


def all_tasks(data_root: Path, output_root: Path, output_dir: str, thinking_tag: str):
    out_base = output_root / output_dir / thinking_tag
    tasks = []

    for hop in HOPS:
        for budget in BUDGETS:
            k = PARALLEL_K[hop][budget]
            tasks.append(
                TaskSpec(
                    name=f"parallel/{hop}/{budget}/{k}sentences",
                    setting="parallel",
                    input_file=data_root / PARALLEL_INPUT.format(hop=hop),
                    output_file=out_base / budget / f"parallel_setting_{hop}.jsonl",
                    k=k,
                    prompt_family="parallel",
                    icl_family="parallel",
                    budget=budget,
                    hop=hop,
                )
            )

    for hop in HOPS:
        for budget in BUDGETS:
            k = CONDITIONAL_K[hop][budget]
            tasks.append(
                TaskSpec(
                    name=f"conditional/{hop}/{budget}/{k}sentences",
                    setting="conditional",
                    input_file=data_root / CONDITIONAL_INPUT.format(hop=hop),
                    output_file=out_base / budget / f"conditional_setting_{hop}.jsonl",
                    k=k,
                    prompt_family="conditional",
                    icl_family="conditional",
                    budget=budget,
                    hop=hop,
                )
            )

    for perspective in PERSPECTIVES:
        for budget in BUDGETS:
            k = SCIENCE_K[perspective][budget]
            tasks.append(
                TaskSpec(
                    name=f"imbalanced/science/{perspective}/{budget}/{k}sentences",
                    setting="imbalanced",
                    input_file=data_root / SCIENCE_INPUT.format(perspective=perspective),
                    output_file=out_base / budget / f"science_domain_{perspective}.jsonl",
                    k=k,
                    prompt_family="diverse_science",
                    icl_family="science",
                    budget=budget,
                    domain="science",
                    perspective=perspective,
                )
            )

    for perspective, budget_map in POLICY_K.items():
        for budget in BUDGETS:
            k = budget_map[budget]
            tasks.append(
                TaskSpec(
                    name=f"imbalanced/policy/{perspective}/{budget}/{k}sentences",
                    setting="imbalanced",
                    input_file=data_root / POLICY_INPUT,
                    output_file=out_base / budget / "policy_domain.jsonl",
                    k=k,
                    prompt_family="diverse_policy",
                    icl_family="policy",
                    budget=budget,
                    domain="policy",
                    perspective=perspective,
                )
            )

    return tasks


def selected(task: TaskSpec, args) -> bool:
    if args.setting != "all" and task.setting != args.setting:
        return False
    if args.budget != "all" and task.budget != args.budget:
        return False
    if args.hop != "all" and task.hop != args.hop:
        return False
    if args.domain != "all" and task.domain != args.domain:
        return False
    if args.perspective != "all" and task.perspective != args.perspective:
        return False
    return True


# --------------------------------------------------------------------------- #
# Request / response
# --------------------------------------------------------------------------- #

GEMMA_THINKING_END = "<channel|>"
GEMMA_THINKING_START = "<|channel>"
GEMMA_FINAL_MARKERS = ("<|channel>final", "<|channel|>final", "<channel>final")
GEMMA_TRAILING_SENTINELS = ("<turn|>", "<eos>")


def message_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(item.get("text", ""))
            elif getattr(item, "type", None) == "text" and getattr(item, "text", None):
                chunks.append(item.text)
        return "".join(chunks)
    return ""


def clean_gemma_sentinels(text: str) -> str:
    text = text.strip()
    for marker in GEMMA_FINAL_MARKERS:
        if text.startswith(marker):
            text = text[len(marker):].lstrip()
    removed = True
    while removed:
        removed = False
        for sentinel in GEMMA_TRAILING_SENTINELS:
            if text.endswith(sentinel):
                text = text[: -len(sentinel)].rstrip()
                removed = True
    return text


def extract_output_text(response, family: FamilySpec) -> str:
    """Drop any thinking prefix and family-specific sentinels."""
    message = response.choices[0].message
    text = message_to_text(getattr(message, "content", ""))

    # rsplit, not split: a degenerate response can emit the closing marker several times
    # (Qwen3-4B does), and splitting on the first one leaves the rest in the output. The
    # answer is whatever follows the LAST close; if that is empty, no answer ever arrived.
    if family.gemma_channels and GEMMA_THINKING_END in text:
        text = text.rsplit(GEMMA_THINKING_END, 1)[1]
    elif "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif family.gemma_channels and GEMMA_THINKING_START in text:
        # Truncated before the final channel opened: nothing usable.
        return ""
    elif "<think>" in text:
        # Qwen3 with thinking on, cut off mid-thought: the answer never arrived. Returning
        # the raw text here would write the hidden reasoning into compressed_context.
        return ""

    if family.gemma_channels:
        text = clean_gemma_sentinels(text)
    return text.strip()


def request_kwargs(args, family: FamilySpec, messages):
    kwargs = {
        "model": args.served_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "timeout": args.request_timeout,
    }
    if family.supports_thinking:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
        }
    return kwargs


async def process_one(idx, entry, client, semaphore, task: TaskSpec, args, family, icl_examples):
    async with semaphore:
        kwargs = request_kwargs(
            args, family, build_messages(entry, task.k, task.prompt_family, icl_examples)
        )
        for attempt in range(args.max_retries + 1):
            try:
                response = await client.chat.completions.create(**kwargs)
                out = dict(entry)
                out["compressed_context"] = extract_output_text(response, family)
                return idx, out
            except Exception as error:
                if attempt == args.max_retries:
                    print(f"Error in {task.name} at index {idx}: {type(error).__name__}: {error}")
                    out = dict(entry)
                    out["compressed_context"] = ERROR_COMPRESSED_CONTEXT
                    return idx, out
                await asyncio.sleep(min(2**attempt, 30) + random.random())


def is_failed(row) -> bool:
    value = row.get("compressed_context")
    return not value or value == ERROR_COMPRESSED_CONTEXT


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def print_task(task: TaskSpec, args):
    status = "exists" if task.output_file.exists() else "new"
    print(
        f"{task.name}\n"
        f"  input:  {task.input_file}\n"
        f"  output: {task.output_file} ({status})\n"
        f"  base_url: {args.base_url}\n"
        f"  model={args.served_model} family={args.family} k={task.k} "
        f"budget={task.budget} enable_thinking={args.enable_thinking}"
    )


async def preflight_model(client, args) -> int | None:
    """Confirm the server is up and serves --served-model; return max_model_len."""
    served = {}
    # Explicit timeout: while the server is still loading weights the port is already
    # listening but requests get no answer, and the client's 10-minute default would
    # look like a hang.
    try:
        page = await client.models.list(timeout=args.request_timeout)
    except Exception as error:
        raise SystemExit(
            "%s did not answer the models endpoint within %.0fs (%s: %s).\n"
            "  The port is open while the server is still loading weights, but HTTP gets no reply;\n"
            "  check that curl %s/models responds, then rerun."
            % (args.base_url, args.request_timeout, type(error).__name__, error, args.base_url.rstrip('/'))
        )
    for model in page.data:
        served[model.id] = getattr(model, "max_model_len", None)
    if args.served_model not in served:
        raise SystemExit(
            f"{args.base_url} does not serve '{args.served_model}'.\n"
            f"  available: {sorted(served)}\n"
            f"  pass --served-model with one of the above."
        )
    max_model_len = served[args.served_model]
    print(
        f"PREFLIGHT: {args.base_url} serves {args.served_model}"
        + (f", max_model_len={max_model_len}" if max_model_len else "")
    )
    return max_model_len


def tokenize_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + "/tokenize"


async def preflight_context(args, tasks, max_model_len: int, probe_rows: int = 5):
    """Check that the longest prompts still leave room for --max-tokens.

    The full ICL prefix is 11k-18k tokens depending on the setting, so a
    server started with too small a --max-model-len fails every request in a
    task with HTTP 400. This check catches that before an output file fills
    with error sentinels.

    Prompt length is probed on the rows with the longest ``context`` strings,
    so the reported number is a close lower bound, not a proven maximum.
    Needs vLLM's /tokenize endpoint; other servers just skip this check.
    """
    url = tokenize_url(args.base_url)
    rows_cache: dict[Path, list] = {}
    measured, problems = [], []

    # A short timeout of its own rather than the generation timeout: if the engine stops
    # answering (/ping still returns 200 while /health and /tokenize hang), the optional
    # check should give up quickly instead of stalling for minutes.
    probe_timeout = min(args.request_timeout, TOKENIZE_PROBE_TIMEOUT)
    async with httpx.AsyncClient(timeout=probe_timeout) as http:
        for task in tasks:
            if task.input_file not in rows_cache:
                rows_cache[task.input_file] = load_jsonl(task.input_file)
            rows = rows_cache[task.input_file]
            if args.limit_per_task is not None:
                rows = rows[: args.limit_per_task]
            probes = sorted(rows, key=lambda r: len(r.get("context", "")), reverse=True)[:probe_rows]
            icl_examples = load_icl_examples(task, args.data_root, args.shuffle_seed)

            worst = 0
            for row in probes:
                payload = {
                    "model": args.served_model,
                    "messages": build_messages(row, task.k, task.prompt_family, icl_examples),
                }
                try:
                    response = await http.post(url, json=payload)
                except httpx.HTTPError as error:
                    print('PREFLIGHT: skipping context check (%s: %s). /tokenize did not answer within %.0fs; '
                          'if /ping responds but /health hangs, the vLLM engine has stopped responding; see the server log.'
                          % (type(error).__name__, error, probe_timeout))
                    return
                if response.status_code != 200:
                    print(f"PREFLIGHT: skipping context check ({url} returned {response.status_code})")
                    return
                worst = max(worst, response.json()["count"])

            needed = worst + args.max_tokens
            measured.append((task.name, worst, needed))
            if needed > max_model_len:
                problems.append((task.name, worst, needed))

    widest = max(len(name) for name, _, _ in measured)
    print(f"  {'task'.ljust(widest)}  {'prompt':>7}  {'+max_tokens':>11}  fits {max_model_len}")
    for name, worst, needed in measured:
        print(f"  {name.ljust(widest)}  {worst:>7}  {needed:>11}  {'yes' if needed <= max_model_len else 'NO'}")

    if problems:
        shortfall = max(needed for _, _, needed in problems)
        raise SystemExit(
            f"\n{len(problems)} task(s) do not fit in max_model_len={max_model_len}.\n"
            f"  Restart the server with --max-model-len {((shortfall + 4095) // 4096) * 4096} or more,\n"
            f"  or lower --max-tokens (needs at least {shortfall - max_model_len} fewer),\n"
            f"  or narrow the run with --setting / --budget.\n"
            f"  Pass --skip-preflight to run anyway."
        )
    print("PREFLIGHT OK: every selected task fits.")


async def run_task(task: TaskSpec, args, family, client):
    existing = None
    if task.output_file.exists():
        if args.retry_errors:
            existing = load_jsonl(task.output_file)
            if len(existing) != _input_row_count(task, args):
                raise SystemExit(
                    f"--retry-errors: {task.output_file} has {len(existing)} rows but the "
                    f"input has {_input_row_count(task, args)}; rerun with --overwrite instead."
                )
            if not any(is_failed(row) for row in existing):
                print(f"SKIP (no failed rows): {task.output_file}")
                return
        elif not args.overwrite:
            print(f"SKIP existing output: {task.output_file}")
            return

    if not task.input_file.exists():
        raise FileNotFoundError(f"Missing input file for {task.name}: {task.input_file}")

    print_task(task, args)
    icl_examples = load_icl_examples(task, args.data_root, args.shuffle_seed)
    rows = load_jsonl(task.input_file)
    if args.limit_per_task is not None:
        rows = rows[: args.limit_per_task]

    if existing is None:
        todo = list(enumerate(rows))
        results = [None] * len(rows)
        print(f"  loaded_rows={len(rows)}, loaded_icl_examples={len(icl_examples)}")
    else:
        results = list(existing[: len(rows)])
        todo = [(idx, rows[idx]) for idx in range(len(rows)) if is_failed(results[idx])]
        print(
            f"  loaded_rows={len(rows)}, loaded_icl_examples={len(icl_examples)}, "
            f"retrying {len(todo)} failed row(s)"
        )

    semaphore = asyncio.Semaphore(args.max_concurrency)
    pending = [
        process_one(idx, entry, client, semaphore, task, args, family, icl_examples)
        for idx, entry in todo
    ]
    for coro in tqdm(asyncio.as_completed(pending), total=len(pending), desc=task.name):
        idx, result = await coro
        results[idx] = result

    failures = sum(1 for row in results if is_failed(row))
    write_jsonl(task.output_file, results)
    print(f"DONE: {task.output_file}" + (f"  ({failures} failed row(s))" if failures else ""))


def _input_row_count(task: TaskSpec, args) -> int:
    n = sum(1 for line in task.input_file.open(encoding="utf-8") if line.strip())
    return n if args.limit_per_task is None else min(n, args.limit_per_task)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ICL context compression over the whole benchmark at 0.5x / 1x / 2x.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-models", action="store_true", help="Print the model registry and exit.")
    parser.add_argument("--model", choices=sorted(MODELS), help="Registry alias; fills in the other model options.")
    parser.add_argument("--served-model", help="Model id the server advertises. Overrides --model.")
    parser.add_argument("--family", choices=sorted(FAMILIES), help="Model family. Overrides --model.")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint. Overrides --model.")
    parser.add_argument("--api-key", default="EMPTY", help="Sent as the bearer token; local vLLM ignores it.")
    parser.add_argument("--output-dir", help="Subdirectory under --output-root. Overrides --model.")

    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--enable-thinking", "--enable_thinking", type=str2bool, default=False,
                        help="Qwen3 / Gemma-4 only; also selects the with_thinking output subdirectory.")
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)

    parser.add_argument("--setting", choices=["all", "parallel", "conditional", "imbalanced"], default="all")
    parser.add_argument("--budget", choices=["all", *BUDGETS], default="all")
    parser.add_argument("--hop", choices=["all", *HOPS], default="all")
    parser.add_argument("--domain", choices=["all", "science", "policy"], default="all")
    parser.add_argument("--perspective", choices=["all", *PERSPECTIVES], default="all")

    parser.add_argument("--limit-per-task", type=int, default=None, help="Only the first N rows; for smoke runs.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute tasks that already have output.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Keep finished rows and re-request only the failed ones.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight and list the selected tasks, then exit without generating.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the served-model and context-length checks.")

    args = parser.parse_args()

    if args.list_models:
        width = max(len(alias) for alias in MODELS)
        print(f"{'alias'.ljust(width)}  {'family'.ljust(7)}  {'served model'.ljust(24)}  output dir")
        for alias in sorted(MODELS):
            spec = MODELS[alias]
            print(f"{alias.ljust(width)}  {spec.family.ljust(7)}  "
                  f"{spec.served_model.ljust(24)}  {spec.output_dir}")
        raise SystemExit(0)

    # --model fills in defaults; the explicit flags win.
    spec = MODELS[args.model] if args.model else None
    for attr, field in (("served_model", "served_model"), ("family", "family"),
                        ("base_url", "base_url"), ("output_dir", "output_dir")):
        if getattr(args, attr) is None:
            if spec is None:
                parser.error(f"--{attr.replace('_', '-')} is required when --model is not given")
            setattr(args, attr, getattr(spec, field))

    if args.overwrite and args.retry_errors:
        parser.error("--overwrite and --retry-errors are mutually exclusive")
    if args.enable_thinking and not FAMILIES[args.family].supports_thinking:
        parser.error(f"family '{args.family}' has no thinking mode; drop --enable-thinking")
    return args


async def main():
    args = parse_args()
    family = FAMILIES[args.family]
    thinking_tag = "with_thinking" if args.enable_thinking else "without_thinking"

    tasks = [
        task
        for task in all_tasks(args.data_root, args.output_root, args.output_dir, thinking_tag)
        if selected(task, args)
    ]
    print(f"Selected {len(tasks)} task(s) for {args.served_model} ({args.family}, {thinking_tag}).")

    missing = [str(task.input_file) for task in tasks if not task.input_file.exists()]
    if missing:
        raise SystemExit("Missing input file(s):\n  " + "\n  ".join(sorted(set(missing))))

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    try:
        # --dry-run also preflights: "will this fit before I start a long run?"
        if not args.skip_preflight:
            max_model_len = await preflight_model(client, args)
            if max_model_len:
                await preflight_context(args, tasks, max_model_len)
        if args.dry_run:
            for task in tasks:
                print_task(task, args)
            return
        for task in tasks:
            await run_task(task, args, family, client)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
