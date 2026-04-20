"""
Annotate subsumption chains for (query, statute) pairs using vLLM.

For each relevant (query, statute) pair in qrels, the LLM identifies:
  - Legal elements (Tatbestandsmerkmale/unsur) in the statute
  - Facts in the query
  - Fact-to-element mappings (which facts satisfy which elements)

Usage:
  1. Start vLLM server:
     vllm serve Qwen/Qwen3.5-9B-Instruct \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.90 \
       --enable-prefix-caching

  2. Run:
     python experiment/annotate_subsumption.py \
       --dataset kuhperdata-humanized \
       --concurrency 8

  Resumes automatically from existing output.
"""

import json
import asyncio
import argparse
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


DATASETS = {
    "kuhperdata-humanized": {"path": "data/kuhperdata-humanized", "lang": "id"},
    "kuhperdata-summarized": {"path": "data/kuhperdata-summarized", "lang": "id"},
    "bsard": {"path": "data/bsard", "lang": "fr"},
    "ilpcsr": {"path": "data/ilpcsr", "lang": "en"},
    "stard": {"path": "data/stard", "lang": "zh"},
}

SYSTEM_PROMPTS = {
    "id": """Anda adalah analis hukum yang mengidentifikasi hubungan subsumsi antara fakta kasus dan unsur-unsur pasal undang-undang.

TUGAS: Diberikan sebuah pertanyaan/kasus hukum dan sebuah pasal yang relevan, identifikasi:
1. Unsur-unsur hukum (legal elements) yang terkandung dalam pasal tersebut
2. Fakta-fakta dalam pertanyaan/kasus
3. Pemetaan: fakta mana yang memenuhi unsur mana

ATURAN:
- Unsur hukum = syarat/kondisi abstrak yang harus dipenuhi agar pasal berlaku
- Fakta = peristiwa/keadaan konkret yang disebutkan dalam pertanyaan
- Setiap fakta bisa memenuhi lebih dari satu unsur
- Jika suatu unsur tidak terpenuhi oleh fakta manapun, tetap cantumkan unsurnya
- Output HARUS berupa JSON valid""",

    "fr": """Vous êtes un analyste juridique qui identifie les relations de subsomption entre les faits d'un cas et les éléments d'un article de loi.

TÂCHE: Étant donné une question/cas juridique et un article de loi pertinent, identifiez:
1. Les éléments juridiques (conditions) contenus dans l'article
2. Les faits dans la question/cas
3. Le mapping: quel fait satisfait quel élément

RÈGLES:
- Élément juridique = condition abstraite qui doit être remplie pour que l'article s'applique
- Fait = événement/circonstance concret mentionné dans la question
- Chaque fait peut satisfaire plusieurs éléments
- Si un élément n'est satisfait par aucun fait, listez-le quand même
- La sortie DOIT être un JSON valide""",

    "en": """You are a legal analyst identifying subsumption relationships between case facts and statutory elements.

TASK: Given a legal question/case and a relevant statute, identify:
1. Legal elements (conditions/requirements) contained in the statute
2. Facts in the question/case
3. Mapping: which facts satisfy which elements

RULES:
- Legal element = abstract condition that must be met for the statute to apply
- Fact = concrete event/circumstance mentioned in the question
- Each fact may satisfy multiple elements
- If an element is not satisfied by any fact, still list the element
- Output MUST be valid JSON""",

    "zh": """你是一位法律分析师，负责识别案件事实与法律条文构成要件之间的涵摄关系。

任务：给定一个法律问题/案例和一个相关法条，识别：
1. 法条中包含的法律要件（构成要件/条件）
2. 问题/案例中的事实
3. 映射：哪些事实满足哪些要件

规则：
- 法律要件 = 法条适用所需满足的抽象条件
- 事实 = 问题中提到的具体事件/情况
- 每个事实可以满足多个要件
- 如果某个要件没有被任何事实满足，仍然列出该要件
- 输出必须是有效的JSON""",
}

USER_PROMPT_TEMPLATES = {
    "id": """Pertanyaan/Kasus:
{query_text}

Pasal yang relevan:
{statute_title}: {statute_text}

Identifikasi unsur-unsur hukum dalam pasal tersebut, fakta-fakta dalam pertanyaan, dan pemetaan subsumsi antara keduanya.

Output dalam format JSON berikut:
{{
  "statute_elements": [
    {{"idx": 0, "text": "deskripsi unsur hukum pertama"}},
    {{"idx": 1, "text": "deskripsi unsur hukum kedua"}}
  ],
  "query_facts": [
    {{"idx": 0, "text": "deskripsi fakta pertama"}},
    {{"idx": 1, "text": "deskripsi fakta kedua"}}
  ],
  "mappings": [
    {{"fact_idx": 0, "element_indices": [0, 1], "reasoning": "penjelasan singkat mengapa fakta ini memenuhi unsur tersebut"}}
  ],
  "unmapped_elements": [1]
}}

Jika pasal terlalu umum atau abstrak sehingga tidak bisa dipecah menjadi unsur-unsur spesifik, gunakan satu unsur saja yang merangkum inti pasal.

JSON:""",

    "fr": """Question/Cas:
{query_text}

Article pertinent:
{statute_title}: {statute_text}

Identifiez les éléments juridiques de l'article, les faits dans la question, et le mapping de subsomption.

Sortie en format JSON suivant:
{{
  "statute_elements": [
    {{"idx": 0, "text": "description du premier élément juridique"}},
    {{"idx": 1, "text": "description du deuxième élément juridique"}}
  ],
  "query_facts": [
    {{"idx": 0, "text": "description du premier fait"}},
    {{"idx": 1, "text": "description du deuxième fait"}}
  ],
  "mappings": [
    {{"fact_idx": 0, "element_indices": [0, 1], "reasoning": "explication brève"}}
  ],
  "unmapped_elements": [1]
}}

Si l'article est trop général pour être décomposé en éléments spécifiques, utilisez un seul élément résumant l'essence de l'article.

JSON:""",

    "en": """Question/Case:
{query_text}

Relevant statute:
{statute_title}: {statute_text}

Identify the legal elements in the statute, facts in the question, and the subsumption mapping between them.

Output in the following JSON format:
{{
  "statute_elements": [
    {{"idx": 0, "text": "description of first legal element"}},
    {{"idx": 1, "text": "description of second legal element"}}
  ],
  "query_facts": [
    {{"idx": 0, "text": "description of first fact"}},
    {{"idx": 1, "text": "description of second fact"}}
  ],
  "mappings": [
    {{"fact_idx": 0, "element_indices": [0, 1], "reasoning": "brief explanation"}}
  ],
  "unmapped_elements": [1]
}}

If the statute is too general to decompose into specific elements, use a single element summarizing the core requirement.

JSON:""",

    "zh": """问题/案例：
{query_text}

相关法条：
{statute_title}：{statute_text}

识别法条中的法律要件、问题中的事实，以及两者之间的涵摄映射。

按以下JSON格式输出：
{{
  "statute_elements": [
    {{"idx": 0, "text": "第一个法律要件的描述"}},
    {{"idx": 1, "text": "第二个法律要件的描述"}}
  ],
  "query_facts": [
    {{"idx": 0, "text": "第一个事实的描述"}},
    {{"idx": 1, "text": "第二个事实的描述"}}
  ],
  "mappings": [
    {{"fact_idx": 0, "element_indices": [0, 1], "reasoning": "简要解释"}}
  ],
  "unmapped_elements": [1]
}}

如果法条过于笼统无法分解为具体要件，使用一个要件概括法条核心要求。

JSON:""",
}


def load_corpus(path):
    corpus = {}
    with open(f"{path}/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = d
    return corpus


def load_queries(path):
    queries = {}
    with open(f"{path}/queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d
    return queries


def load_qrels(filepath):
    qrels = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                if qid not in qrels:
                    qrels[qid] = []
                qrels[qid].append(did)
    return qrels


def load_done(output_path):
    done = set()
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(f"{d['query_id']}_{d['doc_id']}")
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


async def call_llm(client, model, system_prompt, user_prompt, max_tokens=1500):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
    if isinstance(content, list):
        parts = [
            (item.get("text") if isinstance(item, dict) else getattr(item, "text", ""))
            for item in content
        ]
        content = "\n".join(p.strip() for p in parts if p and p.strip())
    return content.strip() if content else "", usage_dict


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


async def annotate_pair(client, model, lang, query_text, statute_title, statute_text, semaphore):
    async with semaphore:
        system_prompt = SYSTEM_PROMPTS[lang]
        user_prompt = USER_PROMPT_TEMPLATES[lang].format(
            query_text=query_text,
            statute_title=statute_title,
            statute_text=statute_text,
        )
        text, usage = await call_llm(client, model, system_prompt, user_prompt)
        parsed = parse_json_response(text)
        return parsed, text, usage


async def run(args):
    cfg = DATASETS[args.dataset]
    data_path = cfg["path"]
    lang = cfg["lang"]

    corpus = load_corpus(data_path)
    queries = load_queries(data_path)

    # Load both train and test qrels
    all_pairs = []
    for split in ["train", "test"]:
        qrels_path = f"{data_path}/qrels_{split}.tsv"
        if not Path(qrels_path).exists():
            continue
        qrels = load_qrels(qrels_path)
        for qid, doc_ids in qrels.items():
            for did in doc_ids:
                all_pairs.append((qid, did, split))

    output_path = f"outputs/subsumption/{args.dataset}/annotations.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output_path)

    pairs_todo = [(qid, did, split) for qid, did, split in all_pairs if f"{qid}_{did}" not in done]
    print(f"Dataset: {args.dataset} ({lang})")
    print(f"Total pairs: {len(all_pairs)}, already done: {len(done)}, remaining: {len(pairs_todo)}")

    if not pairs_todo:
        print("Nothing to do.")
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    model = args.model
    semaphore = asyncio.Semaphore(args.concurrency)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    parse_errors = 0

    async def process_one(qid, did, split):
        nonlocal total_prompt_tokens, total_completion_tokens, parse_errors

        query = queries.get(qid)
        doc = corpus.get(did)
        if not query or not doc:
            return

        parsed, raw_text, usage = await annotate_pair(
            client, model, lang,
            query["text"], doc.get("title", f"Doc {did}"), doc["text"],
            semaphore,
        )
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]

        result = {
            "query_id": qid,
            "doc_id": did,
            "split": split,
            "annotation": parsed,
            "usage": usage,
        }
        if parsed is None:
            result["raw_response"] = raw_text
            parse_errors += 1

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    tasks = [process_one(qid, did, split) for qid, did, split in pairs_todo]
    await tqdm_asyncio.gather(*tasks, desc=f"Annotating {args.dataset}")

    print(f"\nDone! Total tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")
    print(f"Parse errors: {parse_errors}/{len(pairs_todo)}")


def main():
    parser = argparse.ArgumentParser(description="Annotate subsumption chains")
    parser.add_argument("--dataset", default="kuhperdata-humanized",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Instruct")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            args_copy = argparse.Namespace(**vars(args))
            args_copy.dataset = name
            asyncio.run(run(args_copy))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
