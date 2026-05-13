import csv
import json
import re
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances


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


def strip_statute_references(text: str) -> str:
    """Remove statute references from query text to prevent data leakage."""
    _KUHPERDATA = r'(KUHPerdata|KUH\s*Perdata|Kitab\s+Undang-Undang\s+Hukum\s+Perdata)'
    _OTHER_LAWS = r'\s+(?:RBg|HIR|KUHP|KUHAP|UU|UUD|PP|Perpres|Perma|KUHPidana|KUHDagang)\b'
    _PASAL_CHAIN = r'Pasal\s+\d+[a-zA-Z]?(\s*(,|dan|dan\s+Pasal)\s+\d+[a-zA-Z]?)*'

    text = re.sub(_PASAL_CHAIN + r'\s+' + _KUHPERDATA, '', text, flags=re.IGNORECASE)
    text = re.sub(_KUHPERDATA, '', text, flags=re.IGNORECASE)

    def _keep_other_law(m):
        after = m.string[m.end():]
        if re.match(_OTHER_LAWS, after, re.IGNORECASE):
            return m.group(0)
        return ''

    text = re.sub(_PASAL_CHAIN, _keep_other_law, text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def parse_kuhperdata_pasal(law_string: str) -> str | None:
    kuhperdata_patterns = [
        r'KUHPerdata',
        r'Kitab Undang-Undang Hukum Perdata',
        r'KUH\s*Perdata',
    ]
    
    is_kuhperdata = any(re.search(p, law_string, re.IGNORECASE) for p in kuhperdata_patterns)
    
    if not is_kuhperdata:
        return None
    
    # extract pasal number
    pasal_match = re.search(r'Pasal\s+(\d+[a-zA-Z]?)', law_string, re.IGNORECASE)
    if pasal_match:
        return pasal_match.group(1)
    
    return None


def load_queries(json_path: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Load queries from the legacy judgement_to_content.json format."""
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


def load_queries_v2(jsonl_path: str) -> Dict[str, Tuple[List[str], List[List[str]], List[str]]]:
    """Load queries from the vLLM summarizer JSONL format.

    Returns a dict with two query sets:
        {
            "humanized": (queries, ground_truths, case_names),
            "summarized": (queries, ground_truths, case_names),
        }
    Each entry that has valid text and at least one KUHPerdata pasal
    is included. Entries with errors or empty text are skipped.
    """
    results: Dict[str, Tuple[List[str], List[List[str]], List[str]]] = {
        "humanized": ([], [], []),
        "summarized": ([], [], []),
    }

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("error") and rec["error"] != "json_parse_failed":
                continue

            filename = rec.get("filename", "")

            for key, result_key in [("humanized_query", "humanized"),
                                     ("summarized_case", "summarized")]:
                entry = rec.get(key)
                if not isinstance(entry, dict):
                    continue

                text = (entry.get("text") or "").strip()
                if not text:
                    continue

                # Strip any leaked statute references from query text
                text = strip_statute_references(text)
                if not text:
                    continue

                laws = entry.get("relevant_laws", [])
                pasal_numbers = []
                for law in laws:
                    pasal_num = parse_kuhperdata_pasal(law)
                    if pasal_num:
                        pasal_numbers.append(pasal_num)

                if not pasal_numbers:
                    continue

                queries, gts, names = results[result_key]
                queries.append(text)
                gts.append(pasal_numbers)
                names.append(filename)

    return results


def embed_queries_for_splitting(
    queries: List[str],
    model_name: str = "BAAI/bge-m3"
) -> np.ndarray:
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
    n_queries = len(queries)
    n_test = int(n_queries * test_ratio)
    n_clusters = min(n_clusters, n_queries // 5)  # ensure enough samples per cluster
    print(f"Total queries: {n_queries}, Test size: {n_test}, Clusters: {n_clusters}")
    
    embeddings = embed_queries_for_splitting(queries, model_name)
    
    print(f"Clustering {n_queries} queries into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    cluster_indices = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_labels):
        cluster_indices[label].append(idx)
    
    cluster_sizes = {i: len(indices) for i, indices in cluster_indices.items()}
    cluster_centroids = kmeans.cluster_centers_
    
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
    src_dir = Path(__file__).parent
    project_root = src_dir.parent

    return {
        'statute': project_root / 'data' / 'statute' / 'kuh_perdata.csv',
        'queries': project_root / 'data' / 'judgement' / 'judgement_to_content.json',
        'queries_v2': project_root / 'experiment' / 'vllm_summarizer_results.jsonl',
    }


def build_v1(paths: Dict[str, Path], documents, doc_ids):
    """Build dataset from legacy judgement_to_content.json."""
    print("\nLoading queries (v1: judgement_to_content.json)...")
    queries, ground_truths, case_names = load_queries(str(paths['queries']))
    print(f"Loaded {len(queries)} queries with KUHPerdata ground truth")

    if queries:
        print(f"\nSample query: {queries[0][:150]}...")
        print(f"Ground truth pasals: {ground_truths[0]}")

    output_dir = paths['statute'].parent.parent / 'kuhperdata'
    created_files = export_ir_dataset(
        str(output_dir),
        documents, doc_ids,
        queries, ground_truths, case_names,
        test_ratio=0.2,
        n_clusters=50,
        random_state=42
    )
    return created_files


def _export_with_split(
    output_dir: str,
    documents: List[str],
    document_ids: List[str],
    queries: List[str],
    ground_truths: List[List[str]],
    case_names: List[str],
    train_cases: set,
    test_cases: set,
) -> Dict[str, Path]:
    """Export BEIR dataset using a pre-determined case-name split."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created_files = {}

    # Corpus
    corpus_path = output_path / 'corpus.jsonl'
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for doc_id, doc_text in zip(document_ids, documents):
            entry = {'_id': doc_id, 'title': f'Pasal {doc_id}', 'text': doc_text}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    created_files['corpus'] = corpus_path

    # Queries
    queries_path = output_path / 'queries.jsonl'
    with open(queries_path, 'w', encoding='utf-8') as f:
        for i, (query, case_name) in enumerate(zip(queries, case_names)):
            entry = {'_id': f'q{i}', 'text': query, 'metadata': {'case_name': case_name}}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    created_files['queries'] = queries_path

    # Split by case name
    train_indices = [i for i, name in enumerate(case_names) if name in train_cases]
    test_indices = [i for i, name in enumerate(case_names) if name in test_cases]

    # Qrels
    for label, indices, filename in [
        ('qrels', list(range(len(queries))), 'qrels.tsv'),
        ('qrels_train', train_indices, 'qrels_train.tsv'),
        ('qrels_test', test_indices, 'qrels_test.tsv'),
    ]:
        p = output_path / filename
        with open(p, 'w', encoding='utf-8') as f:
            f.write('query_id\tdoc_id\tscore\n')
            for i in indices:
                for doc_id in ground_truths[i]:
                    f.write(f'q{i}\t{doc_id}\t1\n')
        created_files[label] = p

    # Split indices
    split_path = output_path / 'split_indices.json'
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_indices': train_indices,
            'test_indices': test_indices,
            'train_query_ids': [f'q{i}' for i in train_indices],
            'test_query_ids': [f'q{i}' for i in test_indices],
        }, f, indent=2)
    created_files['split_indices'] = split_path

    # Stats
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
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    created_files['stats'] = stats_path

    print(f"  Train: {len(train_indices)} queries, Test: {len(test_indices)} queries")
    return created_files


def build_v2(paths: Dict[str, Path], documents, doc_ids, max_relevant: int = 0):
    """Build dataset from vLLM JSONL with humanized + summarized queries.

    Uses a single case-name-based split so humanized and summarized
    have the same cases in train/test.
    """
    print("\nLoading queries (v2: vllm_summarizer_results.jsonl)...")
    query_sets = load_queries_v2(str(paths['queries_v2']))

    # Validate doc_ids — only keep pasal numbers that exist in corpus
    valid_doc_ids = set(doc_ids)

    for query_type, (queries, ground_truths, case_names) in query_sets.items():
        filtered_queries = []
        filtered_gts = []
        filtered_names = []
        dropped_invalid = 0
        dropped_too_many = 0
        for q, gt, name in zip(queries, ground_truths, case_names):
            valid_gt = [p for p in gt if p in valid_doc_ids]
            if not valid_gt:
                dropped_invalid += 1
                continue
            if max_relevant > 0 and len(valid_gt) > max_relevant:
                dropped_too_many += 1
                continue
            filtered_queries.append(q)
            filtered_gts.append(valid_gt)
            filtered_names.append(name)
        query_sets[query_type] = (filtered_queries, filtered_gts, filtered_names)

        print(f"  {query_type}: {len(filtered_queries)} queries "
              f"({dropped_invalid} dropped — invalid pasal"
              f"{f', {dropped_too_many} dropped — >{max_relevant} relevant docs' if max_relevant > 0 else ''})")

    # Collect all unique case names across both query sets
    all_case_names = set()
    for queries, ground_truths, case_names in query_sets.values():
        all_case_names.update(case_names)
    all_case_names = sorted(all_case_names)

    # Do ONE semantic split on the summarized queries (longer = better clustering)
    # then apply by case name to both datasets
    print(f"\n{'=' * 60}")
    print("Performing shared semantic split (on summarized queries)")
    print(f"{'=' * 60}")

    split_queries, split_gts, split_names = query_sets["summarized"]
    train_indices, test_indices, split_info = semantic_train_test_split(
        queries=split_queries,
        ground_truths=split_gts,
        case_names=split_names,
        test_ratio=0.2,
        n_clusters=50,
        random_state=42,
    )

    train_cases = {split_names[i] for i in train_indices}
    test_cases = {split_names[i] for i in test_indices}
    print(f"  Train cases: {len(train_cases)}, Test cases: {len(test_cases)}")

    # Export both datasets with the same case-level split
    for query_type, (queries, ground_truths, case_names) in query_sets.items():
        if not queries:
            print(f"  Skipping {query_type}: no valid queries")
            continue

        print(f"\n{'=' * 60}")
        print(f"Building BEIR dataset: kuhperdata-{query_type}")
        print(f"{'=' * 60}")

        if queries:
            print(f"Sample query: {queries[0][:150]}...")
            print(f"Ground truth: {ground_truths[0]}")

        output_dir = paths['statute'].parent.parent / f'kuhperdata-{query_type}'
        created_files = _export_with_split(
            str(output_dir),
            documents, doc_ids,
            queries, ground_truths, case_names,
            train_cases, test_cases,
        )

        print(f"\nCreated files for {query_type}:")
        for name, path in created_files.items():
            print(f"  {name}: {path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Build KUHPerdata BEIR dataset")
    parser.add_argument("--version", choices=["v1", "v2", "all"], default="all",
                        help="v1=legacy JSON, v2=vLLM JSONL, all=both")
    parser.add_argument("--max_relevant", type=int, default=10,
                        help="Max relevant docs per query for v2 (0=no limit, default=10)")
    args = parser.parse_args()

    paths = get_default_paths()

    print("Loading statute documents...")
    documents, doc_ids = load_statute_documents(str(paths['statute']))
    print(f"Loaded {len(documents)} documents")

    if args.version in ("v1", "all"):
        if paths['queries'].exists():
            build_v1(paths, documents, doc_ids)
        else:
            print(f"Skipping v1: {paths['queries']} not found")

    if args.version in ("v2", "all"):
        if paths['queries_v2'].exists():
            build_v2(paths, documents, doc_ids, max_relevant=args.max_relevant)
        else:
            print(f"Skipping v2: {paths['queries_v2']} not found")
