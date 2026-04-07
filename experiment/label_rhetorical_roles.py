"""Label query sentences with rhetorical roles using vLLM + Qwen 3.5 9B.

Rhetorical roles follow LegalSeg (Nigam et al., NAACL 2025) / IL-PCSR taxonomy:
  Facts, Issue, Argument by Petitioner, Argument by Respondent,
  Court Reasoning, Precedent, Statute, Conclusion, Court Disclosure, NONE

Usage:
  1. Start vLLM server:
     vllm serve Qwen/Qwen3.5-9B-Instruct --max-model-len 32768 --gpu-memory-utilization 0.90

  2. Run labeling:
     python experiment/label_rhetorical_roles.py --dataset kuhperdata-humanized
     python experiment/label_rhetorical_roles.py --dataset all
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paragnn import DATASETS

SYSTEM_PROMPT = """You are a legal text analysis assistant. Your task is to split legal text into sentences and classify each sentence into a rhetorical role.

Rhetorical Roles:
- Facts: Statements describing the factual background, events, or circumstances of the case
- Issue: The legal question(s) or dispute being addressed
- Argument by Petitioner: Arguments, claims, or demands made by the plaintiff/petitioner/claimant
- Argument by Respondent: Arguments, defenses, or objections made by the defendant/respondent
- Court Reasoning: The court's analysis, interpretation of law, or reasoning process
- Precedent: References to or discussion of prior cases or judicial decisions
- Statute: References to specific laws, articles, regulations, or statutory provisions
- Conclusion: The court's final decision, order, ruling, or judgment
- Court Disclosure: Procedural or administrative statements (dates, parties, jurisdiction)
- NONE: Sentences that do not fit any of the above categories

Output ONLY a JSON array. Each element must have "sentence" and "role" fields."""


LANG_INSTRUCTIONS = {
    "id": "The text is in Indonesian (Bahasa Indonesia). Maintain the original language in the sentence field.",
    "fr": "The text is in French. Maintain the original language in the sentence field.",
    "zh": "The text is in Chinese. Maintain the original language in the sentence field.",
    "en": "The text is in English.",
}


async def label_query(client, model, query_text, lang, max_tokens=2000):
    """Label a single query's sentences with rhetorical roles."""
    lang_note = LANG_INSTRUCTIONS.get(lang, "")
    prompt = f"""{lang_note}

Split the following legal text into sentences and classify each sentence into a rhetorical role.

Legal text:
{query_text}

Output ONLY valid JSON array:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON from response
    try:
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            content = content.rsplit("```", 1)[0]
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, IndexError):
        pass

    # Fallback: return entire text as single NONE paragraph
    return [{"sentence": query_text, "role": "NONE"}]


async def main():
    parser = argparse.ArgumentParser(description="Label query rhetorical roles with LLM")
    parser.add_argument("--dataset", default="kuhperdata-humanized", choices=[*DATASETS, "all"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Instruct")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output_dir", default="outputs/paragnn")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, cfg in datasets.items():
        data_path = cfg["path"]
        lang = cfg["lang"]
        out_dir = f"{args.output_dir}/{name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/rr_labels.json"

        # Load queries
        queries = {}
        with open(f"{data_path}/queries.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                queries[d["_id"]] = d["text"]

        # Load existing results for resume
        existing = {}
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        todo = {qid: text for qid, text in queries.items() if qid not in existing}
        print(f"\n{'='*60}")
        print(f"  Labeling RR: {name} ({lang})")
        print(f"  Total: {len(queries)}, Done: {len(existing)}, Remaining: {len(todo)}")
        print(f"{'='*60}")

        if not todo:
            print("  All queries already labeled.")
            continue

        sem = asyncio.Semaphore(args.concurrency)
        results = dict(existing)

        async def process(qid, text):
            async with sem:
                return qid, await label_query(client, args.model, text, lang)

        tasks = [process(qid, text) for qid, text in todo.items()]
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"  {name}"):
            qid, labels = await coro
            results[qid] = labels

            # Checkpoint every 100
            if len(results) % 100 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        # Final save
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(results)} labels to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
