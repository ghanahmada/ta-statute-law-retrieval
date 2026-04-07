"""Analyze what CatBoost learns from the 1024-d product vector.
Extracts SHAP values and per-dimension contribution patterns.

Usage:
  python experiment/analyze_catboost.py
"""
import numpy as np, json, sys
sys.path.insert(0, 'src')

from catboost import CatBoostClassifier
from util.bm25 import BM25

# Load model
model = CatBoostClassifier()
model.load_model('outputs/jnlp/kuhperdata-humanized/stage1/catboost_model.cbm')

# Load embeddings
corpus_emb = np.load('outputs/jnlp/kuhperdata-humanized/stage1/corpus_embeddings.npy')
query_emb_dict = np.load('outputs/jnlp/kuhperdata-humanized/stage1/query_embeddings.npy', allow_pickle=True).item()
with open('outputs/jnlp/kuhperdata-humanized/stage1/doc_ids.json') as f:
    doc_ids = json.load(f)
doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

# Load test qrels
qrels = {}
with open('data/kuhperdata-humanized/qrels_test.tsv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if i == 0 and parts[0] == 'query_id': continue
        if len(parts) >= 3 and int(parts[2]) > 0:
            qrels.setdefault(parts[0], []).append(parts[1])
qrels = {qid: docs for qid, docs in qrels.items() if len(docs) <= 5}

# Build features for positive + hard negative pairs
features = []
labels = []
pair_info = []

for qid in list(qrels.keys()):
    if qid not in query_emb_dict: continue
    gt = set(qrels[qid])
    q_emb = query_emb_dict[qid]

    # Positive
    for did in gt:
        if did not in doc_id_to_idx: continue
        product = q_emb * corpus_emb[doc_id_to_idx[did]]
        features.append(product)
        labels.append(1)
        pair_info.append({'qid': qid, 'did': did, 'type': 'positive'})

    # Hard negatives (dense top-ranked wrong)
    scores = corpus_emb @ q_emb
    ranked = np.argsort(scores)[::-1]
    count = 0
    for idx in ranked:
        if count >= 5: break
        if doc_ids[idx] in gt: continue
        product = q_emb * corpus_emb[idx]
        features.append(product)
        labels.append(0)
        pair_info.append({'qid': qid, 'did': doc_ids[idx], 'type': 'hard_neg'})
        count += 1

X = np.array(features)
y = np.array(labels)
print(f"Pairs: {len(y)} ({sum(y)} pos, {len(y)-sum(y)} neg)")

# Get CatBoost predictions
probs = model.predict_proba(X)[:, 1]

# How well does it separate?
pos_probs = probs[y == 1]
neg_probs = probs[y == 0]
print(f"\nCatBoost prediction distribution:")
print(f"  Positive pairs: mean={pos_probs.mean():.4f}, median={np.median(pos_probs):.4f}")
print(f"  Negative pairs: mean={neg_probs.mean():.4f}, median={np.median(neg_probs):.4f}")
print(f"  Separation: {pos_probs.mean() - neg_probs.mean():+.4f}")
print(f"  AUC approx: {np.mean(pos_probs[:, None] > neg_probs[None, :]):.4f}")

# SHAP values
print("\nComputing SHAP values...")
shap_values = model.get_feature_importance(type='ShapValues', data=X)
# ShapValues returns [n_samples, n_features + 1], last column is bias
shap_features = shap_values[:, :-1]  # [n_samples, 1024]
shap_bias = shap_values[:, -1]       # [n_samples]

print(f"SHAP shape: {shap_features.shape}")

# Which dimensions contribute POSITIVELY to correct classifications?
pos_shap = shap_features[y == 1]  # SHAP for positive pairs
neg_shap = shap_features[y == 0]  # SHAP for negative pairs

# For positive pairs: which dims push prediction UP (correct)?
# For negative pairs: which dims push prediction DOWN (correct)?
# "Helpful" dim: pushes positive up AND negative down
pos_mean_shap = pos_shap.mean(axis=0)  # mean SHAP contribution per dim for positives
neg_mean_shap = neg_shap.mean(axis=0)

# Discriminative SHAP: dims that push pos up and neg down
disc_shap = pos_mean_shap - neg_mean_shap  # positive = helps distinguish

top_helpful = np.argsort(disc_shap)[::-1][:20]
top_harmful = np.argsort(disc_shap)[:20]

importances = model.get_feature_importance()

print(f"\n=== TOP 20 MOST HELPFUL DIMS (push positive up, negative down) ===")
for rank, idx in enumerate(top_helpful):
    print(f"  Dim {idx:>4}: disc_shap={disc_shap[idx]:+.6f}  "
          f"pos_shap={pos_mean_shap[idx]:+.6f}  neg_shap={neg_mean_shap[idx]:+.6f}  "
          f"importance={importances[idx]:.4f}")

print(f"\n=== TOP 20 MOST HARMFUL DIMS (push negative up, positive down) ===")
for rank, idx in enumerate(top_harmful):
    print(f"  Dim {idx:>4}: disc_shap={disc_shap[idx]:+.6f}  "
          f"pos_shap={pos_mean_shap[idx]:+.6f}  neg_shap={neg_mean_shap[idx]:+.6f}  "
          f"importance={importances[idx]:.4f}")

# Overall: how many dims help vs hurt?
helpful = np.sum(disc_shap > 0)
harmful = np.sum(disc_shap < 0)
print(f"\n=== SUMMARY ===")
print(f"Helpful dims (disc_shap > 0): {helpful}/1024 ({helpful/1024*100:.1f}%)")
print(f"Harmful dims (disc_shap < 0): {harmful}/1024 ({harmful/1024*100:.1f}%)")
print(f"Total helpful SHAP magnitude: {disc_shap[disc_shap > 0].sum():.6f}")
print(f"Total harmful SHAP magnitude: {disc_shap[disc_shap < 0].sum():.6f}")
print(f"Net SHAP (positive means CatBoost succeeds): {disc_shap.sum():+.6f}")

# Correlation between SHAP discriminativeness and feature importance
from scipy.stats import spearmanr
r, p = spearmanr(importances, np.abs(disc_shap))
print(f"\nCorrelation (feature importance vs |disc_shap|): rho={r:.4f}, p={p:.6f}")
print("(Do important features actually help discrimination?)")
