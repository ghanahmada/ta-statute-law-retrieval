"""Test if adding extra features to CatBoost improves over 1024-d product alone.

Ablation:
  A. product only (1024 features) — baseline (JNLP S1)
  B. product + bm25 score (1025 features)
  C. product + bm25 + title_cosine (1026 features)
  D. product + bm25 + sparse_score + title_cosine (1027 features)

Usage:
  python experiment/catboost_multi_feature.py --dataset kuhperdata-humanized
"""
import numpy as np, json, sys, argparse
sys.path.insert(0, 'src')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='kuhperdata-humanized')
parser.add_argument('--max_relevant', type=int, default=5)
args = parser.parse_args()

DATASETS = {
    'kuhperdata-humanized': {'path': 'data/kuhperdata-humanized', 'lang': 'id'},
    'kuhperdata-summarized': {'path': 'data/kuhperdata-summarized', 'lang': 'id'},
    'bsard': {'path': 'data/bsard', 'lang': 'fr'},
    'stard': {'path': 'data/stard', 'lang': 'zh'},
    'ilpcsr': {'path': 'data/ilpcsr', 'lang': 'en'},
}

cfg = DATASETS[args.dataset]
data_path = cfg['path']
lang = cfg['lang']

# Load data
corpus, queries = {}, {}
with open(f'{data_path}/corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line); corpus[d['_id']] = d
with open(f'{data_path}/queries.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line); queries[d['_id']] = d

def load_qrels(path):
    qrels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if i == 0 and parts[0] == 'query_id': continue
            if len(parts) >= 3 and int(parts[2]) > 0:
                qrels.setdefault(parts[0], []).append(parts[1])
    if args.max_relevant > 0:
        qrels = {q: d for q, d in qrels.items() if len(d) <= args.max_relevant}
    return qrels

train_qrels = load_qrels(f'{data_path}/qrels_train.tsv')
test_qrels = load_qrels(f'{data_path}/qrels_test.tsv')

doc_ids = list(corpus.keys())
doc_texts = [corpus[d]['text'] for d in doc_ids]
doc_titles = [corpus[d].get('title', '') for d in doc_ids]
doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

# === Encode with BGE-M3 (all representations) ===
print('Loading BGE-M3...')
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print('Encoding corpus...')
corpus_out = model.encode(doc_texts, batch_size=32, max_length=512,
                          return_dense=True, return_sparse=True, return_colbert_vecs=False)
c_dense = np.array(corpus_out['dense_vecs'])
c_sparse = corpus_out['lexical_weights']

print('Encoding titles...')
title_out = model.encode(doc_titles, batch_size=64, max_length=64,
                         return_dense=True, return_sparse=False, return_colbert_vecs=False)
c_title = np.array(title_out['dense_vecs'])

# BM25
from util.bm25 import BM25
if lang == 'zh':
    import jieba; jieba.setLogLevel(20)
    bm25_texts = [' '.join(jieba.cut(t)) for t in doc_texts]
else:
    bm25_texts = doc_texts
bm25 = BM25(b=0.75, k1=1.5, n_gram=1, lang=lang, use_stemmer=False, use_stopwords=False)
bm25.fit(bm25_texts)

del model  # free GPU memory

# === Build features ===
def encode_queries(query_texts):
    m = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    out = m.encode(query_texts, batch_size=32, max_length=512,
                   return_dense=True, return_sparse=True, return_colbert_vecs=False)
    title_out = m.encode(query_texts, batch_size=32, max_length=64,
                         return_dense=True, return_sparse=False, return_colbert_vecs=False)
    del m
    return np.array(out['dense_vecs']), out['lexical_weights'], np.array(title_out['dense_vecs'])

def build_features(qrels, q_dense, q_sparse, q_title, q_texts, qid_list):
    """Build feature matrices for all configurations."""
    features_A = []  # product only
    features_bm25 = []
    features_sparse = []
    features_title = []
    labels = []

    for qi, qid in enumerate(qid_list):
        gt = set(qrels[qid])
        q_text = q_texts[qi]
        bm25_q = q_text
        if lang == 'zh':
            bm25_q = ' '.join(jieba.cut(q_text))
        bm25_scores = bm25.transform(bm25_q)

        # Positives
        for did in gt:
            if did not in doc_id_to_idx: continue
            di = doc_id_to_idx[did]
            product = q_dense[qi] * c_dense[di]
            features_A.append(product)
            features_bm25.append(bm25_scores[di])
            shared = set(q_sparse[qi].keys()) & set(c_sparse[di].keys())
            features_sparse.append(sum(q_sparse[qi][t] * c_sparse[di][t] for t in shared))
            features_title.append(np.dot(q_title[qi], c_title[di]))
            labels.append(1)

        # Hard negatives (BM25 top-ranked non-relevant)
        ranked_idx = np.argsort(bm25_scores)[::-1]
        neg_count = 0
        for idx in ranked_idx:
            if neg_count >= 10: break
            did = doc_ids[idx]
            if did in gt: continue
            di = idx
            product = q_dense[qi] * c_dense[di]
            features_A.append(product)
            features_bm25.append(bm25_scores[di])
            shared = set(q_sparse[qi].keys()) & set(c_sparse[di].keys())
            features_sparse.append(sum(q_sparse[qi][t] * c_sparse[di][t] for t in shared))
            features_title.append(np.dot(q_title[qi], c_title[di]))
            labels.append(0)
            neg_count += 1

    A = np.array(features_A)
    bm25_col = np.array(features_bm25).reshape(-1, 1)
    sparse_col = np.array(features_sparse).reshape(-1, 1)
    title_col = np.array(features_title).reshape(-1, 1)
    y = np.array(labels)

    configs = {
        'A: product only (1024)': A,
        'B: product + bm25 (1025)': np.hstack([A, bm25_col]),
        'C: product + bm25 + title (1026)': np.hstack([A, bm25_col, title_col]),
        'D: product + bm25 + sparse + title (1027)': np.hstack([A, bm25_col, sparse_col, title_col]),
    }
    return configs, y

# === Train and evaluate ===
from catboost import CatBoostClassifier, Pool
from util.metrics import evaluate_ranking

# Encode train queries
train_qids = [q for q in train_qrels if q in queries]
train_texts = [queries[q]['text'] for q in train_qids]
print(f'Encoding {len(train_texts)} train queries...')
from FlagEmbedding import BGEM3FlagModel
m = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
train_out = m.encode(train_texts, batch_size=32, max_length=512,
                     return_dense=True, return_sparse=True, return_colbert_vecs=False)
train_title = m.encode(train_texts, batch_size=32, max_length=64,
                       return_dense=True, return_sparse=False, return_colbert_vecs=False)
tr_dense = np.array(train_out['dense_vecs'])
tr_sparse = train_out['lexical_weights']
tr_title = np.array(train_title['dense_vecs'])

# Encode test queries
test_qids = [q for q in test_qrels if q in queries]
test_texts = [queries[q]['text'] for q in test_qids]
print(f'Encoding {len(test_texts)} test queries...')
test_out = m.encode(test_texts, batch_size=32, max_length=512,
                    return_dense=True, return_sparse=True, return_colbert_vecs=False)
test_title = m.encode(test_texts, batch_size=32, max_length=64,
                      return_dense=True, return_sparse=False, return_colbert_vecs=False)
te_dense = np.array(test_out['dense_vecs'])
te_sparse = test_out['lexical_weights']
te_title = np.array(test_title['dense_vecs'])
del m

print(f'\nBuilding train features ({len(train_qids)} queries)...')
train_configs, train_y = build_features(train_qrels, tr_dense, tr_sparse, tr_title, train_texts, train_qids)

print(f'Building test features ({len(test_qids)} queries)...')
test_configs, test_y = build_features(test_qrels, te_dense, te_sparse, te_title, test_texts, test_qids)

print(f'\nTrain: {len(train_y)} pairs ({sum(train_y)} pos, {len(train_y)-sum(train_y)} neg)')
print(f'Test:  {len(test_y)} pairs ({sum(test_y)} pos, {len(test_y)-sum(test_y)} neg)')

# Train CatBoost for each config and evaluate retrieval
print(f'\n{"="*70}')
print(f'  ABLATION RESULTS: {args.dataset}')
print(f'{"="*70}')

for config_name, X_train in train_configs.items():
    X_test = test_configs[config_name]

    clf = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6,
                              verbose=False, random_seed=42)
    clf.fit(X_train, train_y)

    # Predict on ALL test query-doc pairs for ranking
    # For proper evaluation, score every query against every corpus doc
    # But that's too expensive. Instead: score test pairs and compute metrics on those.
    test_probs = clf.predict_proba(X_test)[:, 1]

    # Reconstruct per-query rankings from test features
    # We need to map back features to (qid, did) pairs
    # Rebuild the mapping
    rankings = {}
    idx = 0
    for qi, qid in enumerate(test_qids):
        gt = set(test_qrels[qid])
        n_pos = sum(1 for d in gt if d in doc_id_to_idx)
        bm25_q = test_texts[qi]
        if lang == 'zh':
            bm25_q = ' '.join(jieba.cut(bm25_q))
        bm25_scores = bm25.transform(bm25_q)
        ranked_idx = np.argsort(bm25_scores)[::-1]
        n_neg = 0
        neg_dids = []
        for ridx in ranked_idx:
            if n_neg >= 10: break
            if doc_ids[ridx] in gt: continue
            neg_dids.append(doc_ids[ridx])
            n_neg += 1

        # Collect (did, prob) for this query
        cands = [(d, test_probs[idx + i]) for i, d in enumerate(list(gt) + neg_dids) if d in doc_id_to_idx]
        idx += n_pos + len(neg_dids)

        # Sort by probability descending
        cands.sort(key=lambda x: -x[1])
        rankings[qid] = [d for d, _ in cands[:10]]

    # Evaluate
    ground_truth = {qid: list(set(test_qrels[qid])) for qid in test_qids}
    results = evaluate_ranking(rankings, ground_truth, k=10)

    mrr = results[f'mrr@10']
    recall = results[f'recall@10']
    hit = results['hit_rate']
    print(f'  {config_name:<45} MRR={mrr:.4f}  R@10={recall:.4f}  Hit={hit:.1%}')
