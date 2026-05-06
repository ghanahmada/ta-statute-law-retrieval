"""Re-validate RELEVANT candidates from existing expansion logs.

Reads expansion_logs.jsonl, re-checks each RELEVANT candidate with a
skeptical judge prompt (thinking enabled), and writes validated qrels.

Usage:
  1. Start vLLM server (same as expand_qrels.py)

  2. Validate humanized expansion:
     python src/data/validate_expansion.py \
       --expansion_logs data/kuhperdata-exp/expansion_logs.jsonl \
       --dataset kuhperdata-humanized \
       --output_name kuhperdata-exp-v2 \
       --concurrency 8

  Outputs:
    data/{output_name}/qrels_train.tsv   (original + validated expanded)
    data/{output_name}/qrels_test.tsv
    data/{output_name}/validation_log.jsonl  (per-candidate verdicts)

  Resumes automatically from validation_log.jsonl.
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.dataloader import DataLoader


VALIDATOR_PROMPT = """Anda adalah hakim perdata yang skeptis. Periksa apakah pasal "{title}" benar-benar relevan untuk kasus berikut — TOLAK relevansi jika ada satu pun unsur pokok pasal yang tidak hadir dalam fakta kasus. Kasus: "{query_text}" | Pasal: "{article_text}" | Instruksi: sebutkan semua unsur pokok pasal, identifikasi unsur mana yang tidak ada dalam fakta kasus, lalu beri verdict. Jawab dalam satu baris: unsur_pokok: <daftar singkat> | unsur_hilang: <unsur yang tidak ada, atau 'tidak ada'> | verdict: <RELEVAN/TIDAK_RELEVAN>"""


def parse_validator_response(response_text: str) -> bool:
    match = re.search(r'verdict:\s*(RELEVAN|TIDAK_RELEVAN)', response_text)
    if not match:
        return False
    return match.group(1) == "RELEVAN"


async def validate_one(
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    qid: str,
    query_text: str,
    doc_id: str,
    title: str,
    article_text: str,
) -> dict:
    async with sem:
        prompt = VALIDATOR_PROMPT.format(
            title=title,
            query_text=query_text,
            article_text=article_text,
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            msg = response.choices[0].message
            raw = (msg.content or getattr(msg, "reasoning_content", None) or "").strip()
            confirmed = parse_validator_response(raw)
            return {
                "qid": qid,
                "doc_id": doc_id,
                "confirmed": confirmed,
                "raw_response": raw,
                "error": None,
            }
        except Exception as e:
            return {
                "qid": qid,
                "doc_id": doc_id,
                "confirmed": False,
                "raw_response": None,
                "error": str(e),
            }


def load_expansion_logs(path: str) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_done_validations(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    done.add(f"{rec['qid']}_{rec['doc_id']}")
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def write_qrels(qrels: dict[str, dict[str, int]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for qid in sorted(qrels.keys()):
            for did, score in sorted(qrels[qid].items()):
                f.write(f"{qid}\t{did}\t{score}\n")


async def main():
    parser = argparse.ArgumentParser(description="Validate RELEVANT expansion candidates")
    parser.add_argument("--expansion_logs", required=True,
                        help="Path to expansion_logs.jsonl from expand_qrels.py")
    parser.add_argument("--dataset", default="kuhperdata-humanized",
                        help="Original dataset name (for loading corpus + original qrels)")
    parser.add_argument("--output_name", default="kuhperdata-exp-v2",
                        help="Output dataset name under data/")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    src_dir = f"data/{args.dataset}"
    out_dir = f"data/{args.output_name}"
    os.makedirs(out_dir, exist_ok=True)
    val_log_path = f"{out_dir}/validation_log.jsonl"

    # Load corpus
    loader = DataLoader(
        f"{src_dir}/corpus.jsonl",
        f"{src_dir}/queries.jsonl",
        f"{src_dir}/qrels_train.tsv",
    ).load()
    corpus = loader.corpus
    queries = loader.queries

    # Also load test queries
    test_loader = DataLoader(
        f"{src_dir}/corpus.jsonl",
        f"{src_dir}/queries.jsonl",
        f"{src_dir}/qrels_test.tsv",
    ).load()

    # Load expansion logs
    expansion_entries = load_expansion_logs(args.expansion_logs)
    print(f"Loaded {len(expansion_entries)} expansion log entries")

    # Collect all (qid, doc_id) pairs that need validation
    to_validate = []
    for entry in expansion_entries:
        qid = entry["qid"]
        if entry.get("error") or not entry.get("new_relevant"):
            continue
        if qid not in queries:
            continue
        for doc_id in entry["new_relevant"]:
            if doc_id not in corpus:
                continue
            to_validate.append({
                "qid": qid,
                "doc_id": doc_id,
                "query_text": queries[qid]["text"],
                "title": corpus[doc_id]["title"],
                "article_text": corpus[doc_id]["text"],
            })

    print(f"Total RELEVANT candidates to validate: {len(to_validate)}")

    # Resume
    done_keys = load_done_validations(val_log_path)
    pending = [v for v in to_validate if f"{v['qid']}_{v['doc_id']}" not in done_keys]
    if done_keys:
        print(f"Resuming: {len(done_keys)} already validated, {len(pending)} remaining")

    # Validate
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)

    if pending:
        tasks = [
            validate_one(
                client, args.model, sem,
                v["qid"], v["query_text"], v["doc_id"],
                v["title"], v["article_text"],
            )
            for v in pending
        ]

        log_file = open(val_log_path, "a", encoding="utf-8")
        n_confirmed = 0
        n_rejected = 0
        pbar = tqdm_asyncio(total=len(tasks), desc="Validating")

        for coro in asyncio.as_completed(tasks):
            result = await coro
            log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            log_file.flush()
            if result["confirmed"]:
                n_confirmed += 1
            else:
                n_rejected += 1
            pbar.update(1)

        pbar.close()
        log_file.close()
        print(f"\nValidation: {n_confirmed} confirmed, {n_rejected} rejected")

    # Build validated expansion set from full log
    confirmed_pairs: dict[str, set[str]] = {}
    rejected_pairs: dict[str, set[str]] = {}
    with open(val_log_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                qid, doc_id = rec["qid"], rec["doc_id"]
                if rec["confirmed"]:
                    confirmed_pairs.setdefault(qid, set()).add(doc_id)
                else:
                    rejected_pairs.setdefault(qid, set()).add(doc_id)
            except (json.JSONDecodeError, KeyError):
                continue

    total_confirmed = sum(len(v) for v in confirmed_pairs.values())
    total_rejected = sum(len(v) for v in rejected_pairs.values())
    total = total_confirmed + total_rejected
    print(f"\nFinal: {total_confirmed}/{total} confirmed ({100*total_confirmed/max(total,1):.1f}%), "
          f"{total_rejected} rejected ({100*total_rejected/max(total,1):.1f}%)")

    # Build output qrels = original + confirmed expansions
    for split, split_loader in [("train", loader), ("test", test_loader)]:
        qrels = {qid: dict(docs) for qid, docs in split_loader.qrels.items()}
        split_qids = set(qrels.keys())
        n_added = 0
        for qid, doc_ids in confirmed_pairs.items():
            if qid not in split_qids:
                continue
            if qid not in qrels:
                qrels[qid] = {}
            for did in doc_ids:
                if did not in qrels[qid]:
                    qrels[qid][did] = 1
                    n_added += 1
        write_qrels(qrels, f"{out_dir}/qrels_{split}.tsv")
        print(f"  {split}: {n_added} validated expansions added")

    # Copy supporting files
    for fname in ["corpus.jsonl", "queries.jsonl", "split_indices.json"]:
        src = f"{src_dir}/{fname}"
        if os.path.exists(src):
            shutil.copy2(src, f"{out_dir}/{fname}")

    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
