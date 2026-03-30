"""
Batch judgement summarization using vLLM + Qwen 3.5 9B.

Usage:
  1. Start vLLM server:
     vllm serve Qwen/Qwen3.5-9B-Instruct \
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
ATURAN KHUSUS:
1. Gunakan Bahasa Indonesia formal yang lugas.
2. ANTI-BELANDA/LATIN: Jangan gunakan istilah belanda seperti "Verstek", "Niet Ontvankelijke", "Obscuur Libel", "Wanprestasi", "Ex Aequo Et Bono", dsb.
   - Gunakan: "Putusan tanpa kehadiran", "Gugatan tidak dapat diterima", "Gugatan kabur", "Ingkar janji".
3. ANONIMISASI: Gunakan [Penggugat], [Tergugat], [Objek Sengketa], [Nomor Perkara] dan abstraksi lainnya untuk hal-hal yang bersifat sensitif.
4. FOKUS REGULASI: Prioritaskan interpretasi fakta berdasarkan pasal-pasal di KUHPerdata.
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
TUGAS: Anda adalah analis hukum yang sangat teliti. Anda sedang memeriksa BAGIAN #{chunk_id} dari sebuah dokumen putusan.

Konteks global sebagai panduan:
{blueprint}

ATURAN KHUSUS:
1. OBSERVASI LOKAL: Hanya ekstrak informasi yang BENAR-BENAR MUNCUL dalam teks Chunk #{chunk_id} di bawah ini. Jangan mengulang informasi dari Blueprint jika tidak ada buktinya di teks chunk ini.
2. IDENTIFIKASI TAHAPAN: Tentukan apakah chunk ini berisi: 'Identitas Para Pihak', 'Duduk Perkara (Kronologi)', 'Pertimbangan Hukum (Ratio Decidendi)', atau 'Amar Putusan'. Fokuskan ekstraksi pada fungsi bagian tersebut.
3. PENERAPAN GROUND TRUTH: Jika ada pasal yang disebutkan, jangan hanya menyalin isinya. Jelaskan: "Hakim menggunakan pasal ini UNTUK menilai [fakta apa]".

FORMAT OUTPUT WAJIB:
### [A] Analisis Chunk #{chunk_id}
- **Kategori Dokumen**: (Sebutkan: Identitas/Duduk Perkara/Pertimbangan/Amar)
- **Detail Unik**: (Ekstrak fakta spesifik atau argumen yang hanya ada di chunk ini)
- **Logika Hukum**: (Bagaimana pihak atau hakim membangun argumen di bagian ini)

### [B] Temuan Regulasi Spesifik (Jika Ada)
- **Pasal/Doktrin**: [Nomor Pasal]
- **Konteks di Chunk**: [Mengapa pasal ini muncul di sini?]

Dilarang mengeluarkan output "Tidak ada data relevan" kecuali chunk tersebut benar-benar kosong atau hanya berisi disclaimer teknis.

TEKS CHUNK #{chunk_id}:
{chunk_text}
"""
    async with sem:
        text, usage = await call_llm(client, model, prompt, max_tokens=2500)
        return f"\n--- CHUNK {chunk_id} EXTRACTIONS ---\n{text}", usage


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
Identifikasi 3 elemen dari dokumen ini:
1. Subjek & Objek: Siapa [Penggugat], [Tergugat], dan apa [Objek Sengketa]?
2. Garis Besar Konflik: Inti masalah (misal: jual beli, utang piutang).
3. Hasil Akhir: Apakah gugatan dikabulkan, ditolak, atau tidak dapat diterima?

Gunakan informasi ini sebagai konteks untuk proses ekstraksi detail nantinya.
Kembalikan blueprint dalam format poin-poin yang jelas dan terstruktur.

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

    # Phase 3: Synthesizer
    synthesizer_prompt = f"""
Ubah catatan ekstraksi menjadi query yang relevan melalui prosedur berikut.

ATURAN KHUSUS:
1. HAPUS SEMUA ISTILAH BELANDA/LATIN. Jika ada "Verstek", ubah jadi "Putusan tanpa kehadiran". Jika ada "Obscuur Libel", ubah jadi "Gugatan tidak jelas/kabur". Jika ada "NO", ubah jadi "Gugatan tidak dapat diterima".
2. FOKUS KUHPERDATA: Hubungkan temuan fakta dengan pasal-pasal di KUHPerdata
3. BAHASA: jangan gunakan akronim atau istilah yang tidak umum. Gunakan Bahasa Indonesia sebagai abstraksi dari istilah pada notes.
4. Pada bagian Relevansi Pasal, HANYA masukkan pasal yang benar-benar dibahas atau menjadi dasar logika di dalam kalimat query tersebut. Jangan memasukkan semua pasal jika query hanya membahas satu aspek spesifik.
5. Jangan melakukan over-labeling pada Relevansi Pasal. Jika sebuah query membahas tentang kegagalan prosedur atau kejelasan objek, jangan mencantumkan pasal materiil kecuali query tersebut secara eksplisit menanyakan tentang unsur kesalahan atau kerugian.

STRUKTUR OUTPUT:
1. Pertanyaan Hukum Orang Awam: Satu pertanyaan natural dari perspektif non legal. Contoh: "Kenapa saya kalah padahal lawan tidak datang sidang?"
   Relevansi Pasal: [Relevansi: Pasal XXX KUHPerdata, ...].

2. Ringkasan Kasus: Fokus pada Fakta perikatan, tindakan ingkar janji, dan alasan logis hakim menyatakan gugatan kabur karena tuntutan yang saling bertentangan.
   Relevansi Pasal: [Relevansi: Pasal XXX KUHPerdata, ...].

Extracted Notes:
{combined_extractions}
"""
    final_summary, usage = await call_llm(client, model, synthesizer_prompt, max_tokens=2500)
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
                "final_output": None,
                "worker_output": None,
                "usage": None,
            })
            return False

        pages = [p.markdown for p in response.pages]
        try:
            final_output, worker_output, usage = await run_pipeline(
                client, model, pages
            )
            append_result(output_path, {
                "filename": filename,
                "final_output": final_output,
                "worker_output": worker_output,
                "usage": usage,
                "error": None,
            })
            return True
        except Exception as e:
            append_result(output_path, {
                "filename": filename,
                "error": str(e),
                "final_output": None,
                "worker_output": None,
                "usage": None,
            })
            return False


async def main():
    parser = argparse.ArgumentParser(description="Batch vLLM judgement summarizer")
    parser.add_argument("--input_dir", default="experiment/cleaned_downloads")
    parser.add_argument("--output", default="experiment/vllm_summarizer_results.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Instruct")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent PDF pipelines")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N files (0 = all)")
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

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_one(sem, client, args.model, str(p), args.output)
        for p in todo
    ]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing PDFs")
    success = sum(1 for r in results if r)
    failed = len(results) - success
    print(f"\nDone. Success: {success}, Failed: {failed}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
