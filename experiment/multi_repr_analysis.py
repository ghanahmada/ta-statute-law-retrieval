"""Multi-representation independence analysis using BGE-M3 dense/sparse/ColBERT + BM25 + title."""
import json, numpy as np, sys, random
from scipy.stats import spearmanr
sys.path.insert(0, 'src')

def load_data(path):
    corpus, queries, qrels = {}, {}, {}
    with open(f'{path}/corpus.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line); corpus[d['_id']] = d
    with open(f'{path}/queries.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line); queries[d['_id']] = d
    with open(f'{path}/qrels_test.tsv', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if i == 0 and parts[0] == 'query_id': continue
            if len(parts) >= 3 and int(parts[2]) > 0:
                qrels.setdefault(parts[0], []).append(parts[1])
    return corpus, queries, qrels

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split', default='train', choices=['train', 'test'])
parser.add_argument('--neg_type', default='hard', choices=['hard', 'random'])
parser.add_argument('--sample', type=int, default=0, help='0 = use all queries')
args = parser.parse_args()

path = 'data/kuhperdata-humanized'
corpus, queries, _ = load_data(path)
# Load specified split
qrels = {}
with open(f'{path}/qrels_{args.split}.tsv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if i == 0 and parts[0] == 'query_id': continue
        if len(parts) >= 3 and int(parts[2]) > 0:
            qrels.setdefault(parts[0], []).append(parts[1])
qrels = {qid: docs for qid, docs in qrels.items() if len(docs) <= 5}

random.seed(42)
all_qids = [q for q in qrels if q in queries]
sample_qids = random.sample(all_qids, min(args.sample, len(all_qids))) if args.sample > 0 else all_qids
sample_q_texts = [queries[qid]['text'] for qid in sample_qids]
print(f'Split: {args.split}, Neg type: {args.neg_type}, Sample: {len(sample_qids)} queries')
doc_ids = list(corpus.keys())
doc_texts = [corpus[d]['text'] for d in doc_ids]
doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
doc_titles = [corpus[d].get('title', '') for d in doc_ids]

print('Loading BGE-M3...')
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print('Encoding corpus (2127 docs)...')
corpus_out = model.encode(doc_texts, batch_size=32, max_length=512,
                          return_dense=True, return_sparse=True, return_colbert_vecs=True)
print('Encoding queries (30 sample)...')
query_out = model.encode(sample_q_texts, batch_size=32, max_length=512,
                         return_dense=True, return_sparse=True, return_colbert_vecs=True)
print('Encoding titles...')
title_out = model.encode(doc_titles, batch_size=64, max_length=64,
                         return_dense=True, return_sparse=False, return_colbert_vecs=False)
qtitle_out = model.encode(sample_q_texts, batch_size=32, max_length=64,
                          return_dense=True, return_sparse=False, return_colbert_vecs=False)

c_dense = np.array(corpus_out['dense_vecs'])
q_dense = np.array(query_out['dense_vecs'])
c_title = np.array(title_out['dense_vecs'])
q_title = np.array(qtitle_out['dense_vecs'])
c_sparse = corpus_out['lexical_weights']
q_sparse = query_out['lexical_weights']
c_colbert = corpus_out['colbert_vecs']
q_colbert = query_out['colbert_vecs']

print(f'Dense: c={c_dense.shape} q={q_dense.shape}')

from util.bm25 import BM25
bm25 = BM25(b=0.75, k1=1.5, n_gram=1, lang='id', use_stemmer=False, use_stopwords=False)
bm25.fit(doc_texts)

S = {'dense': [], 'sparse': [], 'colbert': [], 'title': [], 'bm25': [], 'label': []}

print(f'Computing scores ({args.neg_type} negatives)...')
for qi, qid in enumerate(sample_qids):
    gt = set(qrels[qid])
    bs = bm25.transform(sample_q_texts[qi])

    if args.neg_type == 'hard':
        # BM25 top-ranked non-relevant as hard negatives
        ranked_idx = np.argsort(bs)[::-1]
        neg_dids = []
        for idx in ranked_idx:
            if len(neg_dids) >= 10: break
            if doc_ids[idx] not in gt:
                neg_dids.append(doc_ids[idx])
        cands = list(gt) + neg_dids
    else:
        cands = list(gt) + random.sample([d for d in doc_ids if d not in gt], 10)

    for did in cands:
        if did not in doc_id_to_idx: continue
        di = doc_id_to_idx[did]
        S['dense'].append(float(np.dot(q_dense[qi], c_dense[di])))
        shared = set(q_sparse[qi].keys()) & set(c_sparse[di].keys())
        S['sparse'].append(float(sum(q_sparse[qi][t] * c_sparse[di][t] for t in shared)))
        sim = np.array(q_colbert[qi]) @ np.array(c_colbert[di]).T
        S['colbert'].append(float(sim.max(axis=1).mean()))
        S['title'].append(float(np.dot(q_title[qi], c_title[di])))
        S['bm25'].append(float(bs[di]))
        S['label'].append(1 if did in gt else 0)

labels = np.array(S['label'])
print(f'Pairs: {len(labels)} ({sum(labels)} pos, {len(labels)-sum(labels)} neg)')

names = ['dense', 'sparse', 'colbert', 'title', 'bm25']
print()
print('=== DISCRIMINATIVE POWER (Spearman with relevance label) ===')
for n in names:
    v = np.array(S[n])
    r, p = spearmanr(v, labels)
    print(f'  {n:<12} rho={r:+.4f}  p={p:.4f}  pos={np.mean(v[labels==1]):.4f}  neg={np.mean(v[labels==0]):.4f}')

print()
print('=== SIGNAL INDEPENDENCE (pairwise Spearman rho) ===')
header = f'{"":>12}' + ''.join(f'{n:>12}' for n in names)
print(header)
for n1 in names:
    row = f'{n1:>12}'
    for n2 in names:
        if n1 == n2:
            row += f'{"--":>12}'
        else:
            r, _ = spearmanr(S[n1], S[n2])
            marker = ' !' if abs(r) > 0.5 else ''
            row += f'{r:>9.3f}{marker:>3}'
    print(row)

print()
print('|rho| < 0.1 = independent (QDER standard)')
print('|rho| > 0.5 = redundant (!)')
