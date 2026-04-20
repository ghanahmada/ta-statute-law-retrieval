"""
Annotate legal elements in statutes and facts in queries using vLLM.

Ground-truth-free: annotations are intrinsic to each document/query,
so they can be used at both training and test time.

Two modes:
  --mode corpus   Annotate statutes: classify each sentence by element type
  --mode queries  Annotate queries: decompose into typed facts

Usage:
  1. Start vLLM server:
     vllm serve Qwen/Qwen3.5-9B-Instruct \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.90 \
       --enable-prefix-caching

  2. Run:
     python experiment/annotate_subsumption.py --dataset kuhperdata-humanized --mode corpus
     python experiment/annotate_subsumption.py --dataset kuhperdata-humanized --mode queries

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

# Fixed taxonomy of element types — will be pre-encoded with BGE-M3 as edge features.
# Designed to be cross-jurisdictional (works for Indonesian, French, English, Chinese law).
ELEMENT_TYPES = [
    "SUBJECT",       # who: person, entity, legal capacity, parties
    "ACTION",        # what: act, omission, conduct, performance
    "OBJECT",        # what is affected: property, rights, obligations, claims
    "CONDITION",     # prerequisite: time, place, manner, threshold
    "CONSEQUENCE",   # result: damage, penalty, obligation, remedy
    "CAUSATION",     # causal link between action and consequence
    "FAULT",         # mental state: intent, negligence, good/bad faith
    "EXCEPTION",     # exclusion, defense, exemption, limitation
    "DEFINITION",    # definitional clause, scope, meaning
    "PROCEDURE",     # procedural requirement, formality, time limit
    "REFERENCE",     # cross-reference to other provisions
    "GENERAL",       # catch-all for sentences that don't fit specific types
]

FACT_TYPES = [
    "CIRCUMSTANCE",  # background situation, context, status
    "ACTION",        # what someone did or failed to do
    "DAMAGE",        # harm, loss, injury suffered
    "DISPUTE",       # what is contested, the legal question
    "GENERAL",       # catch-all
]

# ── System prompts for corpus annotation ────────────────────────────────────

CORPUS_SYSTEM_PROMPTS = {
    "id": """Anda adalah analis hukum. Tugas Anda: menganalisis sebuah pasal undang-undang dan mengidentifikasi unsur-unsur hukum (legal elements) yang terkandung di dalamnya.

Untuk SETIAP kalimat dalam pasal, tentukan tipe unsur hukum dari daftar berikut:
- SUBJECT: siapa (orang, badan hukum, kapasitas hukum, pihak)
- ACTION: apa yang dilakukan (perbuatan, kelalaian, tindakan, pelaksanaan)
- OBJECT: apa yang terdampak (harta, hak, kewajiban, tuntutan)
- CONDITION: prasyarat (waktu, tempat, cara, batas)
- CONSEQUENCE: akibat (kerugian, sanksi, kewajiban, ganti rugi)
- CAUSATION: hubungan sebab-akibat antara perbuatan dan akibat
- FAULT: keadaan mental (kesengajaan, kelalaian, itikad baik/buruk)
- EXCEPTION: pengecualian, pembelaan, pembebasan, pembatasan
- DEFINITION: klausul definisi, ruang lingkup, pengertian
- PROCEDURE: persyaratan prosedural, formalitas, batas waktu
- REFERENCE: rujukan ke pasal/ketentuan lain
- GENERAL: kalimat yang tidak cocok dengan tipe spesifik

ATURAN:
- Pecah pasal menjadi kalimat-kalimat
- Setiap kalimat mendapat TEPAT SATU tipe
- Jika pasal sangat singkat (1 kalimat), tetap pecah menjadi unsur-unsurnya
- Output HARUS berupa JSON valid""",

    "fr": """Vous êtes un analyste juridique. Votre tâche: analyser un article de loi et identifier les éléments juridiques qu'il contient.

Pour CHAQUE phrase de l'article, déterminez le type d'élément juridique parmi:
- SUBJECT: qui (personne, entité, capacité juridique, parties)
- ACTION: quoi (acte, omission, conduite, exécution)
- OBJECT: ce qui est affecté (propriété, droits, obligations, créances)
- CONDITION: prérequis (temps, lieu, manière, seuil)
- CONSEQUENCE: résultat (dommage, sanction, obligation, réparation)
- CAUSATION: lien causal entre action et conséquence
- FAULT: état mental (intention, négligence, bonne/mauvaise foi)
- EXCEPTION: exclusion, défense, exemption, limitation
- DEFINITION: clause définitionnelle, portée, signification
- PROCEDURE: exigence procédurale, formalité, délai
- REFERENCE: renvoi à d'autres dispositions
- GENERAL: phrases ne correspondant à aucun type spécifique

RÈGLES:
- Décomposez l'article en phrases
- Chaque phrase reçoit EXACTEMENT UN type
- La sortie DOIT être un JSON valide""",

    "en": """You are a legal analyst. Your task: analyze a statute and identify the legal elements it contains.

For EACH sentence in the statute, determine the element type from:
- SUBJECT: who (person, entity, legal capacity, parties)
- ACTION: what (act, omission, conduct, performance)
- OBJECT: what is affected (property, rights, obligations, claims)
- CONDITION: prerequisite (time, place, manner, threshold)
- CONSEQUENCE: result (damage, penalty, obligation, remedy)
- CAUSATION: causal link between action and consequence
- FAULT: mental state (intent, negligence, good/bad faith)
- EXCEPTION: exclusion, defense, exemption, limitation
- DEFINITION: definitional clause, scope, meaning
- PROCEDURE: procedural requirement, formality, time limit
- REFERENCE: cross-reference to other provisions
- GENERAL: sentences that don't fit specific types

RULES:
- Split the statute into sentences
- Each sentence gets EXACTLY ONE type
- Output MUST be valid JSON""",

    "zh": """你是一位法律分析师。你的任务：分析一个法条并识别其中包含的法律要件。

对法条中的每个句子，从以下类型中确定要件类型：
- SUBJECT：谁（人、实体、法律能力、当事人）
- ACTION：什么行为（作为、不作为、行为、履行）
- OBJECT：什么受影响（财产、权利、义务、债权）
- CONDITION：前提条件（时间、地点、方式、门槛）
- CONSEQUENCE：结果（损害、处罚、义务、救济）
- CAUSATION：行为与结果之间的因果关系
- FAULT：主观状态（故意、过失、善意/恶意）
- EXCEPTION：排除、抗辩、豁免、限制
- DEFINITION：定义性条款、范围、含义
- PROCEDURE：程序性要求、形式要件、期限
- REFERENCE：对其他条款的引用
- GENERAL：不适合特定类型的句子

规则：
- 将法条分解为句子
- 每个句子获得恰好一个类型
- 输出必须是有效的JSON""",
}

CORPUS_USER_TEMPLATES = {
    "id": """Pasal berikut:
{title}: {text}

Pecah pasal ini menjadi kalimat-kalimat dan klasifikasikan setiap kalimat ke dalam tipe unsur hukum.

Output JSON:
{{
  "sentences": [
    {{"text": "kalimat pertama dari pasal", "element_type": "ACTION"}},
    {{"text": "kalimat kedua dari pasal", "element_type": "CONDITION"}}
  ]
}}

JSON:""",

    "fr": """L'article suivant:
{title}: {text}

Décomposez cet article en phrases et classifiez chaque phrase par type d'élément juridique.

Sortie JSON:
{{
  "sentences": [
    {{"text": "première phrase de l'article", "element_type": "ACTION"}},
    {{"text": "deuxième phrase de l'article", "element_type": "CONDITION"}}
  ]
}}

JSON:""",

    "en": """The following statute:
{title}: {text}

Split this statute into sentences and classify each sentence by legal element type.

Output JSON:
{{
  "sentences": [
    {{"text": "first sentence of the statute", "element_type": "ACTION"}},
    {{"text": "second sentence of the statute", "element_type": "CONDITION"}}
  ]
}}

JSON:""",

    "zh": """以下法条：
{title}：{text}

将此法条分解为句子，并将每个句子按法律要件类型分类。

输出JSON：
{{
  "sentences": [
    {{"text": "法条的第一个句子", "element_type": "ACTION"}},
    {{"text": "法条的第二个句子", "element_type": "CONDITION"}}
  ]
}}

JSON:""",
}

# ── System prompts for query annotation ─────────────────────────────────────

QUERY_SYSTEM_PROMPTS = {
    "id": """Anda adalah analis hukum. Tugas Anda: menganalisis sebuah pertanyaan/kasus hukum dan mendekomposisinya menjadi fakta-fakta yang terpisah.

Untuk SETIAP fakta, tentukan tipenya dari daftar berikut:
- CIRCUMSTANCE: latar belakang situasi, konteks, status
- ACTION: apa yang dilakukan atau tidak dilakukan seseorang
- DAMAGE: kerugian, kehilangan, cedera yang diderita
- DISPUTE: apa yang diperselisihkan, pertanyaan hukumnya
- GENERAL: fakta yang tidak cocok dengan tipe spesifik

ATURAN:
- Setiap fakta adalah satu unit informasi yang berdiri sendiri
- Pertanyaan singkat bisa hanya memiliki 1-2 fakta
- Output HARUS berupa JSON valid""",

    "fr": """Vous êtes un analyste juridique. Votre tâche: analyser une question/cas juridique et le décomposer en faits distincts.

Pour CHAQUE fait, déterminez son type parmi:
- CIRCUMSTANCE: contexte, situation de fond, statut
- ACTION: ce que quelqu'un a fait ou n'a pas fait
- DAMAGE: préjudice, perte, blessure subie
- DISPUTE: ce qui est contesté, la question juridique
- GENERAL: faits ne correspondant à aucun type spécifique

RÈGLES:
- Chaque fait est une unité d'information indépendante
- Les questions courtes peuvent n'avoir que 1-2 faits
- La sortie DOIT être un JSON valide""",

    "en": """You are a legal analyst. Your task: analyze a legal question/case and decompose it into distinct facts.

For EACH fact, determine its type from:
- CIRCUMSTANCE: background situation, context, status
- ACTION: what someone did or failed to do
- DAMAGE: harm, loss, injury suffered
- DISPUTE: what is contested, the legal question
- GENERAL: facts that don't fit specific types

RULES:
- Each fact is one self-contained unit of information
- Short questions may only have 1-2 facts
- Output MUST be valid JSON""",

    "zh": """你是一位法律分析师。你的任务：分析一个法律问题/案例并将其分解为不同的事实。

对每个事实，从以下类型中确定其类型：
- CIRCUMSTANCE：背景情况、上下文、状态
- ACTION：某人做了什么或没做什么
- DAMAGE：遭受的损害、损失、伤害
- DISPUTE：争议焦点、法律问题
- GENERAL：不适合特定类型的事实

规则：
- 每个事实是一个独立的信息单元
- 简短的问题可能只有1-2个事实
- 输出必须是有效的JSON""",
}

QUERY_USER_TEMPLATES = {
    "id": """Pertanyaan/Kasus hukum:
{text}

Dekomposisi menjadi fakta-fakta terpisah dan klasifikasikan setiap fakta.

Output JSON:
{{
  "facts": [
    {{"text": "deskripsi fakta pertama", "fact_type": "CIRCUMSTANCE"}},
    {{"text": "deskripsi fakta kedua", "fact_type": "ACTION"}}
  ]
}}

JSON:""",

    "fr": """Question/Cas juridique:
{text}

Décomposez en faits distincts et classifiez chaque fait.

Sortie JSON:
{{
  "facts": [
    {{"text": "description du premier fait", "fact_type": "CIRCUMSTANCE"}},
    {{"text": "description du deuxième fait", "fact_type": "ACTION"}}
  ]
}}

JSON:""",

    "en": """Legal question/case:
{text}

Decompose into distinct facts and classify each fact.

Output JSON:
{{
  "facts": [
    {{"text": "description of first fact", "fact_type": "CIRCUMSTANCE"}},
    {{"text": "description of second fact", "fact_type": "ACTION"}}
  ]
}}

JSON:""",

    "zh": """法律问题/案例：
{text}

分解为不同的事实并分类每个事实。

输出JSON：
{{
  "facts": [
    {{"text": "第一个事实的描述", "fact_type": "CIRCUMSTANCE"}},
    {{"text": "第二个事实的描述", "fact_type": "ACTION"}}
  ]
}}

JSON:""",
}


# ── Data loading ────────────────────────────────────────────────────────────

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


def load_done_ids(output_path):
    done = set()
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(d["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ── LLM call ───────────────────────────────────────────────────────────────

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
        lines = lines[1:]
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


# ── Corpus annotation ──────────────────────────────────────────────────────

async def annotate_corpus(args):
    cfg = DATASETS[args.dataset]
    data_path = cfg["path"]
    lang = cfg["lang"]

    corpus = load_corpus(data_path)

    output_path = f"outputs/subsumption/{args.dataset}/corpus_elements.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(output_path)

    todo = [(did, doc) for did, doc in corpus.items() if did not in done]
    print(f"Dataset: {args.dataset} ({lang}) — corpus annotation")
    print(f"Total statutes: {len(corpus)}, already done: {len(done)}, remaining: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)
    system_prompt = CORPUS_SYSTEM_PROMPTS[lang]
    template = CORPUS_USER_TEMPLATES[lang]

    total_tokens = [0, 0]
    parse_errors = [0]

    async def process_one(did, doc):
        async with semaphore:
            user_prompt = template.format(
                title=doc.get("title", f"Doc {did}"),
                text=doc["text"],
            )
            text, usage = await call_llm(client, args.model, system_prompt, user_prompt)
            parsed = parse_json_response(text)

            total_tokens[0] += usage["prompt_tokens"]
            total_tokens[1] += usage["completion_tokens"]

            result = {"id": did, "annotation": parsed, "usage": usage}
            if parsed is None:
                result["raw_response"] = text
                parse_errors[0] += 1

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    tasks = [process_one(did, doc) for did, doc in todo]
    await tqdm_asyncio.gather(*tasks, desc=f"Corpus elements ({args.dataset})")

    print(f"\nDone! Tokens: {total_tokens[0]:,} prompt + {total_tokens[1]:,} completion")
    print(f"Parse errors: {parse_errors[0]}/{len(todo)}")


# ── Query annotation ───────────────────────────────────────────────────────

async def annotate_queries(args):
    cfg = DATASETS[args.dataset]
    data_path = cfg["path"]
    lang = cfg["lang"]

    queries = load_queries(data_path)

    output_path = f"outputs/subsumption/{args.dataset}/query_facts.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(output_path)

    todo = [(qid, q) for qid, q in queries.items() if qid not in done]
    print(f"Dataset: {args.dataset} ({lang}) — query annotation")
    print(f"Total queries: {len(queries)}, already done: {len(done)}, remaining: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)
    system_prompt = QUERY_SYSTEM_PROMPTS[lang]
    template = QUERY_USER_TEMPLATES[lang]

    total_tokens = [0, 0]
    parse_errors = [0]

    async def process_one(qid, query):
        async with semaphore:
            user_prompt = template.format(text=query["text"])
            text, usage = await call_llm(client, args.model, system_prompt, user_prompt)
            parsed = parse_json_response(text)

            total_tokens[0] += usage["prompt_tokens"]
            total_tokens[1] += usage["completion_tokens"]

            result = {"id": qid, "annotation": parsed, "usage": usage}
            if parsed is None:
                result["raw_response"] = text
                parse_errors[0] += 1

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    tasks = [process_one(qid, q) for qid, q in todo]
    await tqdm_asyncio.gather(*tasks, desc=f"Query facts ({args.dataset})")

    print(f"\nDone! Tokens: {total_tokens[0]:,} prompt + {total_tokens[1]:,} completion")
    print(f"Parse errors: {parse_errors[0]}/{len(todo)}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Annotate legal elements and facts")
    parser.add_argument("--dataset", default="kuhperdata-humanized",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--mode", required=True, choices=["corpus", "queries", "both"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Instruct")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in datasets:
        args_copy = argparse.Namespace(**vars(args))
        args_copy.dataset = name

        if args.mode in ("corpus", "both"):
            asyncio.run(annotate_corpus(args_copy))
        if args.mode in ("queries", "both"):
            asyncio.run(annotate_queries(args_copy))


if __name__ == "__main__":
    main()
