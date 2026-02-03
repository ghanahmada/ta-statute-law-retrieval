import csv
import json
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any


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


def export_ir_dataset(
    output_dir: str,
    documents: List[str] = None,
    document_ids: List[str] = None,
    queries: List[str] = None,
    ground_truths: List[List[str]] = None,
    case_names: List[str] = None,
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
    
    qrels_path = output_path / 'qrels.tsv'
    with open(qrels_path, 'w', encoding='utf-8') as f:
        f.write('query_id\tdoc_id\tscore\n')
        for i, gt_docs in enumerate(ground_truths):
            query_id = f'q{i}'
            for doc_id in gt_docs:
                # Score 1 = relevant
                f.write(f'{query_id}\t{doc_id}\t1\n')
    created_files['qrels'] = qrels_path
    
    stats_path = output_path / 'dataset_stats.json'
    stats = {
        'num_documents': len(documents),
        'num_queries': len(queries),
        'num_relevance_judgments': sum(len(gt) for gt in ground_truths),
        'avg_relevant_docs_per_query': sum(len(gt) for gt in ground_truths) / len(queries) if queries else 0,
        'avg_query_length_chars': sum(len(q) for q in queries) / len(queries) if queries else 0,
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
    print("Exporting IR dataset...")
    output_dir = paths['statute'].parent.parent / 'ir_dataset'
    created_files = export_ir_dataset(
        str(output_dir),
        documents, doc_ids,
        queries, ground_truths, case_names
    )
    print(f"Created files:")
    for name, path in created_files.items():
        print(f"  {name}: {path}")
