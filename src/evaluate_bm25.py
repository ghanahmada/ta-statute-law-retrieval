import argparse
from typing import List, Tuple, Dict, Set
import numpy as np
from pathlib import Path

from dataset import load_statute_documents, load_queries, get_default_paths
from util.bm25 import BM25
from util.metrics import calculate_mrr, calculate_recall_at_k
from util.dataloader import load_qrels


def evaluate_bm25(
    documents: List[str],
    document_ids: List[str],
    queries: List[str],
    ground_truths: List[List[str]],
    top_k: int = 10,
    bm25_b: float = 0.75,
    bm25_k1: float = 1.5,
    n_gram: int = 1,
) -> Tuple[float, float, List[float], List[float]]:
    print(f"Initializing BM25 (b={bm25_b}, k1={bm25_k1}, n_gram={n_gram})...")
    bm25 = BM25(b=bm25_b, k1=bm25_k1, n_gram=n_gram)
    bm25.fit(documents)
    
    per_query_mrr = []
    per_query_recall = []
    
    print(f"\nEvaluating {len(queries)} queries...")
    for i, (query, gt) in enumerate(zip(queries, ground_truths)):
        scores = bm25.transform(query)
        
        ranked_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [document_ids[idx] for idx in ranked_indices]
        
        mrr = calculate_mrr(ranked_doc_ids, gt, top_k)
        recall = calculate_recall_at_k(ranked_doc_ids, gt, top_k)
        
        per_query_mrr.append(mrr)
        per_query_recall.append(recall)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries...")
    
    mean_mrr = np.mean(per_query_mrr)
    mean_recall = np.mean(per_query_recall)
    
    return mean_mrr, mean_recall, per_query_mrr, per_query_recall


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BM25 retrieval for statute law using MRR'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Number of top documents to consider (default: 10)'
    )
    parser.add_argument(
        '--bm25_b', type=float, default=0.75,
        help='BM25 b parameter (default: 0.75)'
    )
    parser.add_argument(
        '--bm25_k1', type=float, default=1.5,
        help='BM25 k1 parameter (default: 1.5)'
    )
    parser.add_argument(
        '--n_gram', type=int, default=1,
        help='N-gram value for BM25 vectorizer (default: 1)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed results per query'
    )
    parser.add_argument(
        '--use_test_split', action='store_true', default=True,
        help='Evaluate only on test split (default: True)'
    )
    parser.add_argument(
        '--use_all', action='store_true',
        help='Evaluate on all queries (ignores train/test split)'
    )
    
    args = parser.parse_args()
    paths = get_default_paths()
    
    # Determine split mode
    use_test_split = args.use_test_split and not args.use_all
    
    print("=" * 60)
    print("BM25 Statute Law Retrieval Evaluation")
    if use_test_split:
        print("  (Evaluating on TEST split only)")
    else:
        print("  (Evaluating on ALL queries)")
    print("=" * 60)
    
    print("\nLoading statute documents...")
    documents, document_ids = load_statute_documents(str(paths['statute']))
    print(f"  Loaded {len(documents)} documents (pasals)")
    
    print("\nLoading queries...")
    queries, ground_truths, case_names = load_queries(str(paths['queries']))
    print(f"  Loaded {len(queries)} total queries with KUHPerdata ground truth")
    
    if not queries:
        print("\nNo queries with KUHPerdata references found!")
        return
    
    # Filter to test split if requested
    if use_test_split:
        ir_dataset_dir = paths['statute'].parent.parent / 'ir_dataset'
        qrels_test_path = ir_dataset_dir / 'qrels_test.tsv'
        
        if not qrels_test_path.exists():
            print(f"\nWarning: {qrels_test_path} not found. Run 'python src/dataset.py' first.")
            print("Falling back to all queries.")
        else:
            test_qrels = load_qrels(str(qrels_test_path))
            test_query_ids = set(test_qrels.keys())
            
            # Filter queries to only test set
            # Query IDs are "q0", "q1", etc. matching the index
            filtered_queries = []
            filtered_ground_truths = []
            filtered_case_names = []
            
            for i, (q, gt, cn) in enumerate(zip(queries, ground_truths, case_names)):
                query_id = f"q{i}"
                if query_id in test_query_ids:
                    filtered_queries.append(q)
                    filtered_ground_truths.append(gt)
                    filtered_case_names.append(cn)
            
            print(f"  Filtered to {len(filtered_queries)} test queries")
            queries = filtered_queries
            ground_truths = filtered_ground_truths
            case_names = filtered_case_names
    
    print("\n" + "-" * 60)
    mean_mrr, mean_recall, per_query_mrr, per_query_recall = evaluate_bm25(
        documents=documents,
        document_ids=document_ids,
        queries=queries,
        ground_truths=ground_truths,
        top_k=args.top_k,
        bm25_b=args.bm25_b,
        bm25_k1=args.bm25_k1,
        n_gram=args.n_gram,
    )
    
    print("\n" + "=" * 60)
    split_name = "TEST" if use_test_split else "ALL"
    print(f"RESULTS ({split_name} split)")
    print("=" * 60)
    print(f"Top-K: {args.top_k}")
    print(f"BM25 Parameters: b={args.bm25_b}, k1={args.bm25_k1}, n_gram={args.n_gram}")
    print("-" * 60)
    print(f"Mean Reciprocal Rank (MRR@{args.top_k}): {mean_mrr:.4f}")
    print(f"Mean Recall@{args.top_k}: {mean_recall:.4f}")
    print("-" * 60)
    
    # Additional statistics
    non_zero_mrr = [m for m in per_query_mrr if m > 0]
    print(f"Queries with at least one hit in top-{args.top_k}: {len(non_zero_mrr)}/{len(queries)} ({100*len(non_zero_mrr)/len(queries):.1f}%)")
    
    if args.verbose and queries:
        print("\n" + "=" * 60)
        print("DETAILED RESULTS (first 10 queries)")
        print("=" * 60)
        for i in range(min(10, len(queries))):
            print(f"\nQuery {i+1}: {queries[i][:80]}...")
            print(f"  Ground truth pasals: {ground_truths[i]}")
            print(f"  MRR: {per_query_mrr[i]:.4f}, Recall@{args.top_k}: {per_query_recall[i]:.4f}")


if __name__ == '__main__':
    main()
