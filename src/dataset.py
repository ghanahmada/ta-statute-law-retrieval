import csv
import json
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np


def load_statute_documents(csv_path: str) -> Tuple[List[str], List[str]]:
    documents = []
    document_ids = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pasal_nomor = row['pasal_nomor'].strip()
            pasal_text = row['pasal_text'].strip()
            
            doc_text = f"Pasal {pasal_nomor}: {pasal_text}"
            
            documents.append(doc_text)
            document_ids.append(pasal_nomor)
    
    return documents, document_ids


def parse_kuhperdata_pasal(law_string: str) -> str | None:
    kuhperdata_patterns = [
        r'KUHPerdata',
        r'Kitab Undang-Undang Hukum Perdata',
        r'KUH\s*Perdata',
    ]
    
    is_kuhperdata = any(re.search(p, law_string, re.IGNORECASE) for p in kuhperdata_patterns)
    
    if not is_kuhperdata:
        return None
    
    # Extract pasal number
    pasal_match = re.search(r'Pasal\s+(\d+[a-zA-Z]?)', law_string, re.IGNORECASE)
    if pasal_match:
        return pasal_match.group(1)
    
    return None


def load_queries(json_path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = []
    ground_truths = []
    case_names = []
    
    for case_name, case_data in data.items():
        incidents = case_data.get('incidents', [])
        relevant_laws = case_data.get('relevant_laws', [])
        
        # filter only KUHPerdata pasal numbers
        pasal_numbers = []
        for law in relevant_laws:
            pasal_num = parse_kuhperdata_pasal(law)
            if pasal_num:
                pasal_numbers.append(pasal_num)
        
        if not pasal_numbers:
            continue
        
        if not incidents:
            continue
        
        combined_query = ". ".join(incidents)
        
        queries.append(combined_query)
        ground_truths.append(pasal_numbers)
        case_names.append(case_name)
    
    return queries, ground_truths, case_names


def embed_queries_for_splitting(
    queries: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> np.ndarray:
    """
    Embed queries using a small multilingual sentence transformer.
    Used for semantic train/test splitting.
    
    Args:
        queries: List of query texts
        model_name: HuggingFace model name (small multilingual model)
        
    Returns:
        (N, D) numpy array of embeddings
    """
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Embedding {len(queries)} queries...")
    embeddings = model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings


def semantic_train_test_split(
    queries: List[str],
    ground_truths: List[List[str]],
    case_names: List[str],
    test_ratio: float = 0.2,
    n_clusters: int = 10,
    random_state: int = 42,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """
    Split queries into train/test sets using semantic clustering.
    
    Strategy:
    1. Embed all queries using multilingual model
    2. Cluster queries using KMeans
    3. Compute cluster centroids
    4. Select test clusters that are most distant from remaining train clusters
    5. This ensures: train queries are semantically similar within train,
       test queries are semantically similar within test,
       but train and test are semantically distant (no data leakage)
    
    Args:
        queries: List of query texts
        ground_truths: List of ground truth doc lists
        case_names: List of case names
        test_ratio: Fraction of queries for test set
        n_clusters: Number of clusters for KMeans
        random_state: Random seed
        model_name: Embedding model name
        
    Returns:
        train_indices: List of indices for training
        test_indices: List of indices for testing
        split_info: Dict with split statistics
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_distances
    
    n_queries = len(queries)
    n_test = int(n_queries * test_ratio)
    n_clusters = min(n_clusters, n_queries // 5)  # Ensure enough samples per cluster
    
    # Embed queries
    embeddings = embed_queries_for_splitting(queries, model_name)
    
    # Cluster queries
    print(f"Clustering {n_queries} queries into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Get cluster info
    cluster_indices = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_labels):
        cluster_indices[label].append(idx)
    
    cluster_sizes = {i: len(indices) for i, indices in cluster_indices.items()}
    cluster_centroids = kmeans.cluster_centers_
    
    # Compute pairwise distances between cluster centroids
    centroid_distances = cosine_distances(cluster_centroids)
    
    # Greedy selection: pick clusters for test that maximize distance to train
    # Start with all clusters in train, move some to test
    train_clusters = set(range(n_clusters))
    test_clusters = set()
    test_count = 0
    
    while test_count < n_test and train_clusters:
        # For each candidate test cluster, compute min distance to remaining train clusters
        best_cluster = None
        best_min_distance = -1
        
        for candidate in train_clusters:
            remaining_train = train_clusters - {candidate}
            if not remaining_train:
                continue
            
            # Min distance from candidate to any remaining train cluster
            min_dist = min(centroid_distances[candidate, t] for t in remaining_train)
            
            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_cluster = candidate
        
        if best_cluster is None:
            break
        
        # Move this cluster to test
        train_clusters.remove(best_cluster)
        test_clusters.add(best_cluster)
        test_count += cluster_sizes[best_cluster]
    
    # Collect indices
    train_indices = []
    test_indices = []
    
    for cluster_id in train_clusters:
        train_indices.extend(cluster_indices[cluster_id])
    
    for cluster_id in test_clusters:
        test_indices.extend(cluster_indices[cluster_id])
    
    # Compute split statistics
    train_embeddings = embeddings[train_indices]
    test_embeddings = embeddings[test_indices]
    
    # Mean cosine distance between train and test
    if len(train_indices) > 0 and len(test_indices) > 0:
        cross_distances = cosine_distances(train_embeddings, test_embeddings)
        mean_cross_distance = float(np.mean(cross_distances))
        
        # Mean within-train distance
        train_self_distances = cosine_distances(train_embeddings)
        mean_train_distance = float(np.mean(train_self_distances[np.triu_indices(len(train_indices), k=1)]))
        
        # Mean within-test distance
        test_self_distances = cosine_distances(test_embeddings)
        mean_test_distance = float(np.mean(test_self_distances[np.triu_indices(len(test_indices), k=1)]))
    else:
        mean_cross_distance = 0
        mean_train_distance = 0
        mean_test_distance = 0
    
    split_info = {
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "n_clusters": n_clusters,
        "train_clusters": list(train_clusters),
        "test_clusters": list(test_clusters),
        "mean_train_cosine_distance": mean_train_distance,
        "mean_test_cosine_distance": mean_test_distance,
        "mean_train_test_cosine_distance": mean_cross_distance,
        "separation_ratio": mean_cross_distance / max(mean_train_distance, 1e-6),
    }
    
    print(f"\nSemantic Split Results:")
    print(f"  Train: {len(train_indices)} queries ({len(train_clusters)} clusters)")
    print(f"  Test:  {len(test_indices)} queries ({len(test_clusters)} clusters)")
    print(f"  Mean within-train cosine distance: {mean_train_distance:.4f}")
    print(f"  Mean within-test cosine distance:  {mean_test_distance:.4f}")
    print(f"  Mean train-test cosine distance:   {mean_cross_distance:.4f}")
    print(f"  Separation ratio: {split_info['separation_ratio']:.2f}x")
    
    return train_indices, test_indices, split_info


def export_ir_dataset(
    output_dir: str,
    documents: List[str] = None,
    document_ids: List[str] = None,
    queries: List[str] = None,
    ground_truths: List[List[str]] = None,
    case_names: List[str] = None,
    test_ratio: float = 0.2,
    n_clusters: int = 10,
    random_state: int = 42,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> Dict[str, Path]:
    """
    Export IR dataset in BEIR format with semantic train/test split.
    
    Creates:
    - corpus.jsonl: All documents
    - queries.jsonl: All queries
    - qrels.tsv: All relevance judgments (for reference)
    - qrels_train.tsv: Train split relevance judgments
    - qrels_test.tsv: Test split relevance judgments
    - dataset_stats.json: Statistics including split info
    
    The train/test split is based on semantic clustering of queries:
    - Queries in train are semantically similar to each other
    - Queries in test are semantically similar to each other  
    - But train and test queries are semantically distant (maximizes generalization)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if documents is None or document_ids is None:
        paths = get_default_paths()
        documents, document_ids = load_statute_documents(str(paths['statute']))
    
    if queries is None or ground_truths is None or case_names is None:
        paths = get_default_paths()
        queries, ground_truths, case_names = load_queries(str(paths['queries']))
    
    created_files = {}
    
    # Export corpus
    corpus_path = output_path / 'corpus.jsonl'
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for doc_id, doc_text in zip(document_ids, documents):
            entry = {
                '_id': doc_id,
                'title': f'Pasal {doc_id}',
                'text': doc_text
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    created_files['corpus'] = corpus_path
    
    # Export all queries
    queries_path = output_path / 'queries.jsonl'
    with open(queries_path, 'w', encoding='utf-8') as f:
        for i, (query, case_name) in enumerate(zip(queries, case_names)):
            query_id = f'q{i}'
            entry = {
                '_id': query_id,
                'text': query,
                'metadata': {
                    'case_name': case_name
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    created_files['queries'] = queries_path
    
    # Perform semantic train/test split
    print("\n" + "=" * 60)
    print("Performing Semantic Train/Test Split")
    print("=" * 60)
    
    train_indices, test_indices, split_info = semantic_train_test_split(
        queries=queries,
        ground_truths=ground_truths,
        case_names=case_names,
        test_ratio=test_ratio,
        n_clusters=n_clusters,
        random_state=random_state,
        model_name=embedding_model
    )
    
    # Export all qrels (for reference)
    qrels_path = output_path / 'qrels.tsv'
    with open(qrels_path, 'w', encoding='utf-8') as f:
        f.write('query_id\tdoc_id\tscore\n')
        for i, gt_docs in enumerate(ground_truths):
            query_id = f'q{i}'
            for doc_id in gt_docs:
                f.write(f'{query_id}\t{doc_id}\t1\n')
    created_files['qrels'] = qrels_path
    
    # Export train qrels
    qrels_train_path = output_path / 'qrels_train.tsv'
    with open(qrels_train_path, 'w', encoding='utf-8') as f:
        f.write('query_id\tdoc_id\tscore\n')
        for i in train_indices:
            query_id = f'q{i}'
            for doc_id in ground_truths[i]:
                f.write(f'{query_id}\t{doc_id}\t1\n')
    created_files['qrels_train'] = qrels_train_path
    
    # Export test qrels
    qrels_test_path = output_path / 'qrels_test.tsv'
    with open(qrels_test_path, 'w', encoding='utf-8') as f:
        f.write('query_id\tdoc_id\tscore\n')
        for i in test_indices:
            query_id = f'q{i}'
            for doc_id in ground_truths[i]:
                f.write(f'{query_id}\t{doc_id}\t1\n')
    created_files['qrels_test'] = qrels_test_path
    
    # Export split indices for reproducibility
    split_indices_path = output_path / 'split_indices.json'
    with open(split_indices_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_indices': train_indices,
            'test_indices': test_indices,
            'train_query_ids': [f'q{i}' for i in train_indices],
            'test_query_ids': [f'q{i}' for i in test_indices],
        }, f, indent=2)
    created_files['split_indices'] = split_indices_path
    
    # Export stats
    stats_path = output_path / 'dataset_stats.json'
    stats = {
        'num_documents': len(documents),
        'num_queries': len(queries),
        'num_train_queries': len(train_indices),
        'num_test_queries': len(test_indices),
        'num_relevance_judgments': sum(len(gt) for gt in ground_truths),
        'num_train_judgments': sum(len(ground_truths[i]) for i in train_indices),
        'num_test_judgments': sum(len(ground_truths[i]) for i in test_indices),
        'avg_relevant_docs_per_query': sum(len(gt) for gt in ground_truths) / len(queries) if queries else 0,
        'avg_query_length_chars': sum(len(q) for q in queries) / len(queries) if queries else 0,
        'split_info': split_info,
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    created_files['stats'] = stats_path
    
    return created_files


def get_default_paths() -> Dict[str, Path]:
    # assuming this file is in src/
    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    
    return {
        'statute': project_root / 'data' / 'statute' / 'kuh_perdata.csv',
        'queries': project_root / 'data' / 'judgement' / 'judgement_to_content.json',
    }


if __name__ == '__main__':
    paths = get_default_paths()
    
    print("Loading statute documents...")
    documents, doc_ids = load_statute_documents(str(paths['statute']))
    print(f"Loaded {len(documents)} documents")
    print(f"Sample document: {documents[0][:100]}...")
    
    print("\nLoading queries (filtered for KUHPerdata only)...")
    queries, ground_truths, case_names = load_queries(str(paths['queries']))
    print(f"Loaded {len(queries)} queries with KUHPerdata ground truth")
    
    if queries:
        print(f"\nSample query (combined incidents): {queries[0][:150]}...")
        print(f"Ground truth pasals: {ground_truths[0]}")
        print(f"Case name: {case_names[0]}")
    
    print("\n" + "=" * 60)
    print("Exporting IR dataset with Semantic Train/Test Split...")
    print("=" * 60)
    output_dir = paths['statute'].parent.parent / 'ir_dataset'
    created_files = export_ir_dataset(
        str(output_dir),
        documents, doc_ids,
        queries, ground_truths, case_names,
        test_ratio=0.2,  # 20% test
        n_clusters=50,   # More clusters = finer split (closer to exact 80/20)
        random_state=42
    )
    
    print("\nCreated files:")
    for name, path in created_files.items():
        print(f"  {name}: {path}")
    
    print("\n" + "=" * 60)
    print("Usage:")
    print("  - qrels_train.tsv: Use for training (semantically grouped)")
    print("  - qrels_test.tsv:  Use for evaluation (unseen, distant queries)")
    print("  - split_indices.json: Reproducible split indices")
    print("=" * 60)
