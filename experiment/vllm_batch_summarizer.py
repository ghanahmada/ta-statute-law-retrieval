"""
Batch judgement summarization using vLLM + Qwen 3.6 27B FP8.

Usage:
  1. Start vLLM server:
     vllm serve QuantTrio/Qwen3.6-27B-AWQ \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.90 \
       --enable-prefix-caching

  2. Run this script:
     python experiment/vllm_batch_summarizer.py \
       --input_dir experiment/cleaned_downloads \
       --output experiment/vllm_summarizer_results.jsonl \
       --concurrency 8

  Resumes automatically from the last checkpoint (reads existing output JSONL).
"""

import os
import re
import json
import fitz
import asyncio
import argparse
import subprocess
from openai import AsyncOpenAI
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from types import SimpleNamespace

# ── PDF parsing ──────────────────────────────────────────────────────────────

def _sanitize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def parse_pdf_text(filepath: str):
    try:
        with open(filepath, "rb") as f:
            file_bytes = f.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [
            SimpleNamespace(markdown=_sanitize_text(page.get_text("text")))
            for page in doc
        ]
        return SimpleNamespace(pages=pages)
    except Exception as e:
        print(f"[WARN] Error parsing {Path(filepath).name}: {e}")
        return None


# ── LLM call ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Anda adalah analis hukum perdata Indonesia. Tugas Anda adalah mengekstrak insiden hukum dari dokumen putusan pengadilan dan memetakannya ke pasal-pasal KUHPerdata (Kitab Undang-Undang Hukum Perdata) yang relevan.

ATURAN:
1. HANYA fokus pada pasal KUHPerdata. Abaikan UUD, UU, KUHP, KUHAP, dan regulasi lainnya.
2. Jika dokumen TIDAK membahas KUHPerdata sama sekali, kembalikan JSON kosong: {"incidents": [], "relevant_laws": []}
3. Gunakan Bahasa Indonesia formal. Ganti istilah Belanda/Latin dengan padanan Indonesia (Wanprestasi → Ingkar janji, Verstek → Putusan tanpa kehadiran, dll).
4. Anonimkan pihak: gunakan [Penggugat], [Tergugat], [Objek Sengketa], dst.
5. Setiap incident adalah ringkasan fakta hukum yang berdiri sendiri, bukan opini atau analisis.
""".strip()


async def call_llm(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 1500,
) -> tuple[str, dict]:
    """Returns (text, usage_dict)."""
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
    usage = response.usage
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }

    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip(), usage_dict
    if isinstance(content, list):
        parts = [
            (item.get("text") if isinstance(item, dict) else getattr(item, "text", ""))
            for item in content
        ]
        return "\n".join(p.strip() for p in parts if p and p.strip()), usage_dict
    return "", usage_dict


# ── KUHPerdata regex filter ──────────────────────────────────────────────────

_KUHPERDATA_PATTERN = re.compile(
    r'Pasal\s+(\d+[a-zA-Z]?)'
    r'(\s*(,|dan|dan\s+Pasal|jo\.?\s*Pasal)\s+(\d+[a-zA-Z]?))*'
    r'\s+(KUHPerdata|KUH\s*Perdata|Kitab\s+Undang-Undang\s+Hukum\s+Perdata)',
    re.IGNORECASE,
)

_PASAL_NUMBER = re.compile(r'Pasal\s+(\d+[a-zA-Z]?)', re.IGNORECASE)


def extract_kuhperdata_pasal(text: str) -> list[str]:
    """Extract unique KUHPerdata pasal references from text using regex."""
    pasal_set = set()
    for match in _KUHPERDATA_PATTERN.finditer(text):
        # Extract all pasal numbers from the matched span
        span = match.group(0)
        for num_match in _PASAL_NUMBER.finditer(span):
            pasal_set.add(f"Pasal {num_match.group(1)} KUHPerdata")
    return sorted(pasal_set, key=lambda x: int(re.search(r'\d+', x).group()))


# ── Pipeline: Blueprint → Workers → Synthesizer ─────────────────────────────

async def worker_chunk(
    client: AsyncOpenAI,
    model: str,
    chunk_id: int,
    chunk_text: str,
    blueprint: str,
    sem: asyncio.Semaphore,
) -> tuple[str, dict]:
    prompt = f"""
Anda memeriksa BAGIAN #{chunk_id} dari sebuah dokumen putusan pengadilan.

Konteks global:
{blueprint}

TUGAS: Dari chunk ini, ekstrak:
1. Semua fakta hukum / insiden (kronologi, tindakan para pihak, kerugian, objek sengketa, argumen hakim)
2. Semua pasal atau regulasi yang disebutkan (catat apa adanya, jangan filter)

ATURAN:
- Catat SEMUA fakta hukum meskipun tidak menyebut pasal apapun.
- Catat SEMUA referensi regulasi yang muncul apa adanya.
- Ganti istilah Belanda/Latin: Wanprestasi → Ingkar janji, Verstek → Putusan tanpa kehadiran, Onrechtmatige daad → Perbuatan melawan hukum.
- Anonimkan: [Penggugat], [Tergugat], [Objek Sengketa].

FORMAT OUTPUT:
FAKTA: [daftar fakta hukum yang ditemukan di chunk ini]
REGULASI: [daftar semua pasal/regulasi yang dirujuk di chunk ini]

TEKS CHUNK #{chunk_id}:
{chunk_text}
"""
    async with sem:
        text, usage = await call_llm(client, model, prompt, max_tokens=1500)
        return f"\n--- CHUNK {chunk_id} ---\n{text}", usage


async def run_pipeline(
    client: AsyncOpenAI,
    model: str,
    pages: list[str],
    worker_batch_size: int = 10,
    chunk_size_words: int = 10000,
    overlap_words: int = 200,
) -> tuple[str, str, dict]:
    """Returns (final_summary, worker_extractions, total_usage)."""
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    full_text = " ".join(pages)
    words = full_text.split()

    # Phase 1: Blueprint
    planner_context = "\n".join(pages[:5] + pages[-5:])
    blueprint_prompt = f"""
Identifikasi secara singkat dari dokumen putusan ini:
1. Siapa [Penggugat] dan [Tergugat]?
2. Apa inti sengketa (jual beli, utang piutang, waris, tanah, dll)?
3. Regulasi apa saja yang dirujuk?
4. Hasil akhir: gugatan dikabulkan, ditolak, atau tidak dapat diterima?

Jawab dalam poin-poin singkat.

Document text: {planner_context}
"""
    blueprint, usage = await call_llm(client, model, blueprint_prompt, max_tokens=1500)
    total_usage["prompt_tokens"] += usage["prompt_tokens"]
    total_usage["completion_tokens"] += usage["completion_tokens"]

    # Phase 2: Workers
    step = chunk_size_words - overlap_words
    chunks = [" ".join(words[i : i + chunk_size_words]) for i in range(0, len(words), step)]

    sem = asyncio.Semaphore(worker_batch_size)
    tasks = [
        worker_chunk(client, model, idx + 1, chunk_text, blueprint, sem)
        for idx, chunk_text in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks)

    worker_texts = []
    for text, usage in results:
        worker_texts.append(text)
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
    combined_extractions = "\n".join(worker_texts)

    # Phase 2.5: Regex-extract KUHPerdata pasal from raw PDF + worker output
    kuhperdata_pasal = extract_kuhperdata_pasal(full_text + "\n" + combined_extractions)

    if not kuhperdata_pasal:
        # No KUHPerdata found — skip synthesizer, save tokens
        empty = json.dumps({
            "humanized_query": {"text": "", "relevant_laws": []},
            "summarized_case": {"text": "", "relevant_laws": []},
        })
        return empty, combined_extractions, total_usage

    pasal_list = ", ".join(kuhperdata_pasal)

    # Phase 3: Synthesizer — scoped to KUHPerdata facts only
    synthesizer_prompt = f"""
Berdasarkan hasil ekstraksi berikut, telah ditemukan rujukan pasal KUHPerdata:
{pasal_list}

Tugas Anda adalah menyusun dua keluaran:
1. Pertanyaan dalam bahasa sehari-hari (humanized_query)
2. Ringkasan kasus (summarized_case)

KETENTUAN UMUM:
- Fokuskan analisis hanya pada prinsip hukum perdata yang tercermin dari pasal KUHPerdata.
- Abaikan konteks lain di luar hukum perdata, termasuk namun tidak terbatas pada:
  pengujian undang-undang, hukum pidana, hukum tata negara, dan kebijakan publik.
- Jika suatu perkara berasal dari konteks non-perdata, lakukan abstraksi sehingga
  permasalahan dapat dipahami sebagai sengketa antar subjek hukum dalam ranah perdata.

KETENTUAN KHUSUS:
1. "humanized_query"
   - Ditulis dalam 1 kalimat singkat, bahasa sehari-hari.
   - Menggambarkan inti persoalan dari sudut pandang orang awam.
   - Tidak menggunakan istilah teknis hukum.

2. "summarized_case"
   - Ditulis dalam 3–5 kalimat.
   - Berisi kronologi singkat yang menekankan hubungan hukum, hak, dan kewajiban para pihak.
   - Tidak menyebut konteks institusi atau rezim hukum di luar perdata.

3. Larangan dalam teks:
   - Tidak menyebut nama lembaga negara atau proses pengujian undang-undang.
   - Tidak menggunakan istilah pidana atau konteks non-perdata lainnya.

FORMAT OUTPUT (JSON saja):
{{
  "humanized_query": {{
    "text": "...",
    "relevant_laws": ["Pasal XXX KUHPerdata"]
  }},
  "summarized_case": {{
    "text": "...",
    "relevant_laws": ["Pasal XXX KUHPerdata"]
  }}
}}

Catatan:
- "relevant_laws" harus diambil dari daftar pasal KUHPerdata di atas.
- Pastikan output selalu berupa JSON valid.

Hasil ekstraksi:
{combined_extractions}
"""
    final_summary, usage = await call_llm(client, model, synthesizer_prompt, max_tokens=2000)
    total_usage["prompt_tokens"] += usage["prompt_tokens"]
    total_usage["completion_tokens"] += usage["completion_tokens"]

    return final_summary, combined_extractions, total_usage


# ── Batch runner ─────────────────────────────────────────────────────────────

def load_done(output_path: str) -> set[str]:
    """Load already-processed filenames from the output JSONL."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["filename"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_result(output_path: str, record: dict) -> None:
    """Append a single result to the output JSONL (atomic per-file checkpoint)."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def process_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    filepath: str,
    output_path: str,
) -> bool:
    """Process a single PDF. Returns True on success."""
    async with sem:
        filename = Path(filepath).name
        response = parse_pdf_text(filepath)
        if response is None:
            append_result(output_path, {
                "filename": filename,
                "error": "pdf_parse_failed",
                "humanized_query": None,
                "summarized_case": None,
                "raw_llm_output": None,
                "usage": None,
            })
            return False

        pages = [p.markdown for p in response.pages]
        try:
            raw_output, worker_extractions, usage = await run_pipeline(
                client, model, pages
            )
            # Try to parse JSON from LLM output
            parsed = None
            try:
                # Strip markdown code fences if present
                clean = raw_output.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                    clean = clean.rsplit("```", 1)[0]
                parsed = json.loads(clean)
            except (json.JSONDecodeError, IndexError):
                pass

            # Post-process: ensure relevant_laws only contains KUHPerdata
            if parsed:
                kuhp_re = re.compile(r'Pasal\s+\d+[a-zA-Z]?\s+(KUHPerdata|KUH\s*Perdata)', re.IGNORECASE)
                for key in ("humanized_query", "summarized_case"):
                    entry = parsed.get(key, {})
                    if isinstance(entry, dict) and "relevant_laws" in entry:
                        entry["relevant_laws"] = [
                            law for law in entry["relevant_laws"]
                            if kuhp_re.search(law)
                        ]

            record = {
                "filename": filename,
                "humanized_query": parsed.get("humanized_query") if parsed else None,
                "summarized_case": parsed.get("summarized_case") if parsed else None,
                "raw_llm_output": raw_output if parsed is None else None,
                "usage": usage,
                "error": None if parsed else "json_parse_failed",
            }
            append_result(output_path, record)
            return parsed is not None
        except Exception as e:
            append_result(output_path, {
                "filename": filename,
                "error": str(e),
                "humanized_query": None,
                "summarized_case": None,
                "raw_llm_output": None,
                "usage": None,
            })
            return False


def upload_to_hf(output_path: str, hf_repo: str) -> bool:
    """Upload the JSONL results file to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo="extraction/vllm_summarizer_results.jsonl",
            repo_id=hf_repo,
            repo_type="dataset",
        )
        count = sum(1 for line in open(output_path, "r", encoding="utf-8") if line.strip())
        print(f"\n[Checkpoint] Uploaded {count} results to HF: {hf_repo}")
        return True
    except Exception as e:
        print(f"\n[Checkpoint] HF upload failed: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Batch vLLM judgement summarizer")
    parser.add_argument("--input_dir", default="experiment/cleaned_downloads")
    parser.add_argument("--output", default="experiment/vllm_summarizer_results.jsonl")
    parser.add_argument("--model", default="QuantTrio/Qwen3.6-27B-AWQ")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent PDF pipelines")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N files (0 = all)")
    parser.add_argument("--hf_repo", default="ghanahmada/kuhperdata",
                        help="HuggingFace repo for checkpoint uploads")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Upload to HF every N completed PDFs")
    args = parser.parse_args()

    # Connect to vLLM
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    # Discover PDFs
    pdf_files = sorted(Path(args.input_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {args.input_dir}")

    # Resume from checkpoint
    done = load_done(args.output)
    todo = [p for p in pdf_files if p.name not in done]
    print(f"Already done: {len(done)}, remaining: {len(todo)}")

    if args.limit > 0:
        todo = todo[: args.limit]
        print(f"Limited to {args.limit} files")

    if not todo:
        print("Nothing to process.")
        return

    # Process with periodic HF checkpoint uploads
    sem = asyncio.Semaphore(args.concurrency)
    completed = 0
    since_last_upload = 0
    success = 0
    failed = 0
    pbar = None

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(todo), desc="Processing PDFs")
    except ImportError:
        pass

    async def process_and_count(filepath):
        nonlocal completed, since_last_upload, success, failed
        result = await process_one(sem, client, args.model, filepath, args.output)
        completed += 1
        since_last_upload += 1
        if result:
            success += 1
        else:
            failed += 1
        if pbar:
            pbar.update(1)

        # Checkpoint upload
        if since_last_upload >= args.checkpoint_every:
            since_last_upload = 0
            upload_to_hf(args.output, args.hf_repo)

    tasks = [process_and_count(str(p)) for p in todo]
    await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    # Final upload
    print(f"\nDone. Success: {success}, Failed: {failed}")
    print(f"Results saved to: {args.output}")
    upload_to_hf(args.output, args.hf_repo)
    print("Final checkpoint uploaded to HuggingFace.")


if __name__ == "__main__":
    asyncio.run(main())
