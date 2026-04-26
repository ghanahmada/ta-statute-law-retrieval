"""Expand qrels using BM25 top-50 candidates + LLM subsumption judgment.

For each query, batches all 50 BM25 candidates into a single LLM call.
The LLM judges whether each statute article's legal elements (unsur-unsur)
are invoked by the case facts described in the query.

Outputs BEIR-format dataset with original + expanded ground truth merged.

Usage:
  1. Start vLLM server:
     vllm serve Qwen/Qwen3.6-27B-FP8 \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.90 \
       --enable-prefix-caching

  2. Expand humanized:
     python src/data/expand_qrels.py \
       --dataset kuhperdata-humanized \
       --output_name kuhperdata-exp \
       --top_k 50 \
       --concurrency 8

  3. Expand summarized:
     python src/data/expand_qrels.py \
       --dataset kuhperdata-summarized \
       --output_name kuhperdata-summ-exp \
       --top_k 50 \
       --concurrency 8

  4. (Optional) Expand humanized with reformulated queries:
     python src/data/expand_qrels.py \
       --dataset kuhperdata-humanized \
       --output_name kuhperdata-exp \
       --source_queries data/kuhperdata-exp/queries_reformulated.jsonl \
       --top_k 50 \
       --concurrency 8

  Output (BEIR format):
    data/{output_name}/corpus.jsonl          (copied from source)
    data/{output_name}/queries.jsonl         (copied or reformulated)
    data/{output_name}/qrels_train.tsv       (original + expanded)
    data/{output_name}/qrels_test.tsv        (original + expanded)
    data/{output_name}/split_indices.json    (copied from source)
    data/{output_name}/expansion_log.jsonl   (raw LLM judgments for audit)

  Resumes automatically from expansion_log.jsonl.
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.bm25 import BM25
from util.dataloader import DataLoader


SYSTEM_PROMPT = """Anda adalah ahli hukum perdata Indonesia yang menggunakan metode analisis unsur untuk menilai relevansi pasal.

Metode analisis unsur:
1. Identifikasi unsur-unsur pokok dari pasal (elemen hukum yang harus terpenuhi)
2. Periksa apakah SETIAP unsur pokok dibahas atau terpenuhi oleh fakta kasus
3. Jika seluruh unsur pokok terpenuhi/dibahas oleh fakta → RELEVAN
4. Jika ada unsur pokok yang sama sekali tidak ada dalam fakta → TIDAK_RELEVAN

Aturan penting:
- "Dibahas" berarti fakta kasus menyentuh unsur tersebut, termasuk jika hakim menolak — yang penting unsur tersebut menjadi pokok pembahasan
- Kesamaan kata kunci BUKAN dasar relevansi. Contoh: kata "mengembalikan" muncul di kasus pinjam-pakai dan di pasal sewa-menyewa, tapi jika hubungan hukumnya berbeda (pinjam ≠ sewa), maka pasal sewa TIDAK RELEVAN
- Pasal umum (1365, 1320, 1338) harus dianalisis seketat pasal khusus — jangan otomatis menandai RELEVAN hanya karena pasal tersebut sering muncul"""


def build_judgment_prompt(query_text: str, candidates: list[dict]) -> str:
    articles_text = ""
    for i, cand in enumerate(candidates, 1):
        articles_text += f"\n[{i}] {cand['title']}\n{cand['text']}\n"

    return f"""Skenario kasus:
\"\"\"{query_text}\"\"\"

Untuk setiap pasal di bawah, lakukan analisis unsur lalu tentukan RELEVAN atau TIDAK_RELEVAN.
{articles_text}
FORMAT JAWABAN (per pasal):
[nomor] unsur: (sebutkan unsur-unsur pokok pasal secara singkat) | fakta: (unsur mana yang terpenuhi/tidak oleh kasus) | RELEVAN/TIDAK_RELEVAN

Contoh:
[1] unsur: perbuatan, melawan hukum, kerugian, kesalahan, kausalitas | fakta: semua unsur dibahas dalam kasus | RELEVAN
[2] unsur: hubungan sewa-menyewa, pengembalian barang sewaan | fakta: kasus tentang pinjam-pakai bukan sewa | TIDAK_RELEVAN"""


def parse_judgments(response_text: str, n_candidates: int) -> list[bool]:
    results = [False] * n_candidates
    for line in response_text.strip().split("\n"):
        line = line.strip()
        m = re.match(r'\[(\d+)\]', line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n_candidates:
                has_relevan = bool(re.search(r'(?<!\w)RELEVAN(?!\w)', line))
                has_tidak = bool(re.search(r'TIDAK_RELEVAN', line))
                results[idx] = has_relevan and not has_tidak
    return results


async def judge_one_query(
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    qid: str,
    query_text: str,
    candidates: list[dict],
) -> dict:
    async with sem:
        prompt = build_judgment_prompt(query_text, candidates)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8092,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = response.choices[0].message.content.strip()
            judgments = parse_judgments(raw, len(candidates))
            new_relevant = [
                cand["doc_id"]
                for cand, is_rel in zip(candidates, judgments)
                if is_rel
            ]
            return {
                "qid": qid,
                "raw_response": raw,
                "new_relevant": new_relevant,
                "candidate_doc_ids": [c["doc_id"] for c in candidates],
                "n_candidates": len(candidates),
                "n_new": len(new_relevant),
                "error": None,
            }
        except Exception as e:
            return {
                "qid": qid,
                "raw_response": None,
                "new_relevant": [],
                "candidate_doc_ids": [c["doc_id"] for c in candidates],
                "n_candidates": len(candidates),
                "n_new": 0,
                "error": str(e),
            }


def compute_bm25_topk(
    loader: DataLoader, query_ids: list[str], top_k: int, lang: str,
) -> dict[str, list[str]]:
    doc_ids, doc_texts = loader.get_corpus_texts()

    if lang == "zh":
        import jieba
        jieba.setLogLevel(20)
        bm25_texts = [" ".join(jieba.cut(t)) for t in doc_texts]
    else:
        bm25_texts = doc_texts

    bm25 = BM25(b=0.75, k1=1.5, n_gram=1, lang=lang,
                 use_stemmer=False, use_stopwords=False)
    bm25.fit(bm25_texts)

    query_texts = [loader.queries[qid]["text"] for qid in query_ids]
    if lang == "zh":
        query_texts = [" ".join(jieba.cut(t)) for t in query_texts]

    print(f"Computing BM25 top-{top_k} for {len(query_ids)} queries...")
    scores = bm25.transform(query_texts)

    topk_per_query = {}
    for i, qid in enumerate(query_ids):
        row = scores[i].toarray().flatten() if hasattr(scores[i], 'toarray') else np.array(scores[i]).flatten()
        top_indices = np.argsort(row)[::-1][:top_k]
        topk_per_query[qid] = [doc_ids[idx] for idx in top_indices]

    return topk_per_query


def write_qrels(qrels: dict[str, dict[str, int]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for qid in sorted(qrels.keys()):
            for did, score in sorted(qrels[qid].items()):
                f.write(f"{qid}\t{did}\t{score}\n")


def load_done(log_path: str) -> tuple[set[str], dict[str, list[str]]]:
    done = set()
    expansions = {}
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["qid"])
                    if rec.get("new_relevant"):
                        expansions[rec["qid"]] = rec["new_relevant"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return done, expansions


async def expand_split(
    split_name: str,
    loader: DataLoader,
    topk_per_query: dict[str, list[str]],
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    log_path: str,
    done_qids: set[str],
) -> dict[str, list[str]]:
    sem = asyncio.Semaphore(concurrency)
    corpus = loader.corpus

    tasks = []
    for qid in topk_per_query:
        if qid not in loader.queries:
            continue
        if qid not in loader.qrels:
            continue
        if qid in done_qids:
            continue

        existing_relevant = {
            did for did, s in loader.qrels.get(qid, {}).items() if s > 0
        }
        candidates = []
        for did in topk_per_query[qid]:
            if did in existing_relevant:
                continue
            if did not in corpus:
                continue
            candidates.append({
                "doc_id": did,
                "title": corpus[did]["title"],
                "text": corpus[did]["text"],
            })

        if not candidates:
            continue

        query_text = loader.queries[qid]["text"]
        tasks.append(
            judge_one_query(client, model, sem, qid, query_text, candidates)
        )

    if not tasks:
        print(f"  {split_name}: nothing to process.")
        return {}

    print(f"  {split_name}: judging {len(tasks)} queries ({concurrency} concurrent)...")
    results = await tqdm_asyncio.gather(*tasks, desc=f"Expanding {split_name}")

    expansions = {}
    with open(log_path, "a", encoding="utf-8") as f:
        for result in results:
            result["split"] = split_name
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result["new_relevant"]:
                expansions[result["qid"]] = result["new_relevant"]

    return expansions


async def main():
    parser = argparse.ArgumentParser(description="Expand qrels via LLM subsumption")
    parser.add_argument("--dataset", default="kuhperdata-humanized")
    parser.add_argument("--output_name", default="kuhperdata-exp",
                        help="Output dataset name under data/")
    parser.add_argument("--source_queries", default=None,
                        help="Path to reformulated queries.jsonl (optional)")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_relevant", type=int, default=5)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    src_dir = f"data/{args.dataset}"
    out_dir = f"data/{args.output_name}"
    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/expansion_log.jsonl"

    # Copy corpus
    shutil.copy2(f"{src_dir}/corpus.jsonl", f"{out_dir}/corpus.jsonl")

    # Copy or use reformulated queries
    queries_src = args.source_queries or f"{src_dir}/queries.jsonl"
    shutil.copy2(queries_src, f"{out_dir}/queries.jsonl")
    if args.source_queries:
        print(f"Using reformulated queries from {args.source_queries}")

    # Load both splits
    train_loader = DataLoader(
        f"{src_dir}/corpus.jsonl",
        f"{out_dir}/queries.jsonl",
        f"{src_dir}/qrels_train.tsv",
    ).load()
    test_loader = DataLoader(
        f"{src_dir}/corpus.jsonl",
        f"{out_dir}/queries.jsonl",
        f"{src_dir}/qrels_test.tsv",
    ).load()

    if args.max_relevant > 0:
        train_loader.filter_max_relevant(args.max_relevant)
        test_loader.filter_max_relevant(args.max_relevant)

    # Detect language
    lang_map = {
        "kuhperdata-humanized": "id", "kuhperdata-summarized": "id",
        "bsard": "fr", "ilpcsr": "en", "stard": "zh",
    }
    lang = lang_map.get(args.dataset, "id")

    # BM25 top-k for all queries
    all_qids = sorted(set(train_loader.qrels.keys()) | set(test_loader.qrels.keys()))
    topk = compute_bm25_topk(train_loader, all_qids, args.top_k, lang)

    # Resume
    done_qids, prev_expansions = load_done(log_path)
    if done_qids:
        print(f"Resuming: {len(done_qids)} queries already judged")

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    # --- Train split ---
    train_qids_set = set(train_loader.qrels.keys())
    train_topk = {qid: docs for qid, docs in topk.items() if qid in train_qids_set}

    new_train = await expand_split(
        "train", train_loader, train_topk,
        client, args.model, args.concurrency, log_path, done_qids,
    )

    # Build expanded train qrels
    train_qrels = {qid: dict(docs) for qid, docs in train_loader.qrels.items()}
    for qid, new_docs in {**prev_expansions, **new_train}.items():
        if qid not in train_qids_set:
            continue
        if qid not in train_qrels:
            train_qrels[qid] = {}
        for did in new_docs:
            if did not in train_qrels[qid]:
                train_qrels[qid][did] = 1

    # Refresh done set
    done_qids, prev_expansions_all = load_done(log_path)

    # --- Test split ---
    test_qids_set = set(test_loader.qrels.keys())
    test_topk = {qid: docs for qid, docs in topk.items() if qid in test_qids_set}

    new_test = await expand_split(
        "test", test_loader, test_topk,
        client, args.model, args.concurrency, log_path, done_qids,
    )

    # Build expanded test qrels
    test_qrels = {qid: dict(docs) for qid, docs in test_loader.qrels.items()}
    for qid, new_docs in {**prev_expansions_all, **new_test}.items():
        if qid not in test_qids_set:
            continue
        if qid not in test_qrels:
            test_qrels[qid] = {}
        for did in new_docs:
            if did not in test_qrels[qid]:
                test_qrels[qid][did] = 1

    # Write
    write_qrels(train_qrels, f"{out_dir}/qrels_train.tsv")
    write_qrels(test_qrels, f"{out_dir}/qrels_test.tsv")

    if os.path.exists(f"{src_dir}/split_indices.json"):
        shutil.copy2(f"{src_dir}/split_indices.json", f"{out_dir}/split_indices.json")

    # Stats
    def count_pos(qrels):
        return sum(1 for docs in qrels.values() for s in docs.values() if s > 0)
    def unique_docs(qrels):
        return len({did for docs in qrels.values() for did, s in docs.items() if s > 0})
    def doc_freq(qrels):
        freq = {}
        for docs in qrels.values():
            for did, s in docs.items():
                if s > 0:
                    freq[did] = freq.get(did, 0) + 1
        return freq

    orig_train = count_pos(train_loader.qrels)
    orig_test = count_pos(test_loader.qrels)
    exp_train = count_pos(train_qrels)
    exp_test = count_pos(test_qrels)

    print(f"\n{'='*60}")
    print(f"  Expansion Results: {args.dataset} -> {args.output_name}")
    print(f"{'='*60}")
    print(f"  Train pairs: {orig_train} -> {exp_train} (+{exp_train - orig_train})")
    print(f"  Train docs:  {unique_docs(train_loader.qrels)} -> {unique_docs(train_qrels)}")
    print(f"  Test pairs:  {orig_test} -> {exp_test} (+{exp_test - orig_test})")
    print(f"  Test docs:   {unique_docs(test_loader.qrels)} -> {unique_docs(test_qrels)}")
    print(f"  Avg relevant/query (train): {orig_train/len(train_loader.qrels):.1f} -> {exp_train/len(train_qrels):.1f}")
    print(f"  Output: {out_dir}/")

    # Hub vs non-hub analysis
    orig_all_qrels = {}
    for qid, docs in train_loader.qrels.items():
        orig_all_qrels.setdefault(qid, {}).update(docs)
    for qid, docs in test_loader.qrels.items():
        orig_all_qrels.setdefault(qid, {}).update(docs)
    exp_all_qrels = {}
    for qid, docs in train_qrels.items():
        exp_all_qrels.setdefault(qid, {}).update(docs)
    for qid, docs in test_qrels.items():
        exp_all_qrels.setdefault(qid, {}).update(docs)

    orig_freq = doc_freq(orig_all_qrels)
    exp_freq = doc_freq(exp_all_qrels)

    hub_threshold = np.percentile(list(orig_freq.values()), 95)
    hub_docs = {did for did, f in orig_freq.items() if f >= hub_threshold}
    non_hub_docs = {did for did in exp_freq if did not in hub_docs}

    added_pairs = {}
    for qid, docs in exp_all_qrels.items():
        orig_docs = orig_all_qrels.get(qid, {})
        for did, s in docs.items():
            if s > 0 and did not in orig_docs:
                added_pairs[did] = added_pairs.get(did, 0) + 1

    hub_added = sum(c for did, c in added_pairs.items() if did in hub_docs)
    non_hub_added = sum(c for did, c in added_pairs.items() if did not in hub_docs)
    new_docs_added = {did for did in added_pairs if did not in orig_freq}

    print(f"\n{'='*60}")
    print(f"  Hub vs Non-Hub Analysis")
    print(f"{'='*60}")
    print(f"  Hub threshold: docs with >= {hub_threshold:.0f} queries in original")
    print(f"  Hub docs: {len(hub_docs)}  |  Non-hub docs: {len(non_hub_docs)}")
    print(f"  Added pairs to hub docs:     {hub_added}")
    print(f"  Added pairs to non-hub docs: {non_hub_added}")
    print(f"  Newly appearing docs (not in original ground truth): {len(new_docs_added)}")
    print(f"  Hub/non-hub add ratio: {hub_added/(non_hub_added or 1):.2f}")

    top_added = sorted(added_pairs.items(), key=lambda x: -x[1])[:20]
    print(f"\n  Top 20 most-added articles:")
    print(f"  {'Article':<15} {'Added':>6} {'Orig Freq':>10} {'Type':<8}")
    print(f"  {'-'*45}")
    for did, count in top_added:
        orig_f = orig_freq.get(did, 0)
        typ = "HUB" if did in hub_docs else "non-hub"
        title = train_loader.corpus.get(did, {}).get("title", did)
        print(f"  {title:<15} {count:>6} {orig_f:>10} {typ:<8}")


if __name__ == "__main__":
    asyncio.run(main())
