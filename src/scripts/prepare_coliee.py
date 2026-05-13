"""Convert COLIEE 2024 statute data (raw XML + text) to BEIR format.

Input directory layout (--raw_dir):
  train/riteval_{year}_en.xml   — labeled pairs (H18–R04)
  text/civil_code_en-1to724-2-processed.txt  — Civil Code corpus

Output (data/coliee/):
  corpus.jsonl        768 Civil Code articles
  queries.jsonl       all bar-exam statement pairs as queries
  qrels_train.tsv     Y-labeled pairs from H18–R03
  qrels_test.tsv      Y-labeled pairs from R04 (most recent year)

Each pair_id (e.g. R04-01-I) is one query; its t1 article is the
relevant doc when label=Y.

Usage:
  python src/scripts/prepare_coliee.py --raw_dir "C:/path/to/COLIEE2024statute_data-English"
"""
import argparse
import json
import re
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

TEST_YEARS = {"R03", "R04"}
OUTPUT_DIR = Path("data/coliee")


# ---------------------------------------------------------------------------
# Corpus parsing
# ---------------------------------------------------------------------------

def parse_corpus(processed_txt: Path) -> list[dict]:
    articles = []
    hierarchy = {k: None for k in ("part", "chapter", "section", "subsection", "division")}

    with open(processed_txt, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Part "):
            hierarchy["part"] = " ".join(line.split()[2:])
            for k in ("chapter", "section", "subsection", "division"):
                hierarchy[k] = None
        elif line.startswith("Chapter "):
            hierarchy["chapter"] = " ".join(line.split()[2:])
            for k in ("section", "subsection", "division"):
                hierarchy[k] = None
        elif line.startswith("Section "):
            hierarchy["section"] = " ".join(line.split()[2:])
            for k in ("subsection", "division"):
                hierarchy[k] = None
        elif line.startswith("Subsection "):
            hierarchy["subsection"] = " ".join(line.split()[2:])
            hierarchy["division"] = None
        elif line.startswith("Division "):
            hierarchy["division"] = " ".join(line.split()[2:])
        elif line == "<ARTICLE>":
            i += 1
            title_raw = lines[i].strip()
            title = title_raw.strip("()") if title_raw.startswith("(") else title_raw
            if title.lower() == "none":
                title = ""
            i += 1
            art_line = lines[i].strip()
            m = re.match(r"Article\s+([\d]+(?:-\d+)*)", art_line)
            if not m:
                i += 1
                continue
            art_no = m.group(1)
            content_lines = []
            while i < len(lines) and lines[i].strip():
                content_lines.append(lines[i].rstrip())
                i += 1
            text = " ".join(content_lines)
            meta = {k: v for k, v in hierarchy.items() if v}
            meta["article_no"] = art_no
            articles.append({
                "_id": f"article_{art_no}",
                "title": title,
                "text": text,
                "metadata": meta,
            })

        i += 1

    return articles


# ---------------------------------------------------------------------------
# Train XML processing
# ---------------------------------------------------------------------------

def extract_article_no(t1_text: str) -> str | None:
    m = re.search(r"Article\s+([\d]+(?:-\d+)*)", t1_text)
    return m.group(1) if m else None


def process_train_files(
    train_dir: Path,
    test_years: set[str],
) -> tuple[dict, dict, dict]:
    """Return (queries, train_qrels, test_qrels).

    queries     : pair_id -> t2 text
    train_qrels : pair_id -> article_no  (Y-labeled, non-test years)
    test_qrels  : pair_id -> article_no  (Y-labeled, test years)
    """
    queries: dict[str, str] = {}
    train_qrels: dict[str, str] = {}
    test_qrels: dict[str, str] = {}

    for xml_file in sorted(train_dir.glob("riteval_*_en.xml")):
        year = xml_file.stem.replace("riteval_", "").replace("_en", "")
        is_test = year in test_years

        tree = ET.parse(xml_file)
        for pair in tree.getroot().findall(".//pair"):
            pair_id = pair.get("id")
            label = pair.get("label")
            t1_el = pair.find("t1")
            t2_el = pair.find("t2")
            if t1_el is None or t2_el is None:
                continue

            t2_text = (t2_el.text or "").strip()
            queries[pair_id] = t2_text

            if label == "Y":
                art_no = extract_article_no((t1_el.text or ""))
                if art_no:
                    if is_test:
                        test_qrels[pair_id] = art_no
                    else:
                        train_qrels[pair_id] = art_no

    return queries, train_qrels, test_qrels


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_jsonl(rows: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):>6} rows -> {path}")


def write_qrels(qrels: dict[str, str], path: Path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "score"])
        for qid, art_no in sorted(qrels.items()):
            writer.writerow([qid, f"article_{art_no}", 1])
    print(f"  Wrote {len(qrels):>6} rows -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        default=r"C:\Users\ghana\Downloads\COLIEE2024statute_data-English (2)",
        help="Path to unpacked COLIEE 2024 statute data directory",
    )
    parser.add_argument(
        "--test_years", nargs="+", default=sorted(TEST_YEARS),
        help="Year(s) to hold out as test set (default: R03 R04)",
    )
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    test_years = set(args.test_years)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing Civil Code corpus...")
    articles = parse_corpus(raw_dir / "text" / "civil_code_en-1to724-2-processed.txt")
    print(f"  {len(articles)} articles parsed")

    print(f"Processing train XML files (test_years={sorted(test_years)})...")
    queries, train_qrels, test_qrels = process_train_files(raw_dir / "train", test_years)
    print(f"  {len(queries)} queries | {len(train_qrels)} train positives | {len(test_qrels)} test positives")

    print("Writing output files...")
    write_jsonl(articles, out_dir / "corpus.jsonl")
    write_jsonl(
        [{"_id": qid, "text": text} for qid, text in sorted(queries.items())],
        out_dir / "queries.jsonl",
    )
    write_qrels(train_qrels, out_dir / "qrels_train.tsv")
    write_qrels(test_qrels, out_dir / "qrels_test.tsv")

    print("\nDone.")
    print(f"  Corpus  : {len(articles)} articles")
    print(f"  Queries : {len(queries)} (all pairs including N-labeled)")
    print(f"  Train   : {len(train_qrels)} positive pairs")
    print(f"  Test    : {len(test_qrels)} positive pairs (years {sorted(test_years)})")


if __name__ == "__main__":
    main()
