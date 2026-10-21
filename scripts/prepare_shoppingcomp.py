"""Convert ShoppingComp Chinese tasks into RAG documents and evaluation sets."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "ecommerce" / "shoppingcomp" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ecommerce" / "shoppingcomp"

GT_FILE = "ShoppingComp_97_zh.jsonl"
SAFETY_FILE = "ShoppingComp_traps_48_zh.jsonl"
SOURCE_URL = "https://hf-mirror.com/datasets/huaixiao/ShoppingComp"
LICENSE = "MIT"

URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
URL_TRAILING_CHARS = ".,;:!?)]}，。；：！？）】"
TRACKING_QUERY_KEYS = {
    "gclid",
    "fbclid",
    "msclkid",
    "yclid",
    "igshid",
    "spm",
    "scm",
    "sourcefrom",
    "sr_share",
}

OFFICIAL_DOMAINS = {
    "apple.com",
    "asus.com",
    "bosch-home.com",
    "casarte.com",
    "dell.com",
    "dessmann.com",
    "dyson.com",
    "electrolux.com",
    "fotile.com",
    "haier.com",
    "honor.com",
    "hp.com",
    "huawei.com",
    "kaadas.com",
    "lenovo.com",
    "lg.com",
    "lockin.com",
    "midea.com",
    "mi.com",
    "miele.com",
    "oppo.com",
    "panasonic.com",
    "philips.com",
    "samsung.com",
    "sony.com",
    "sony.net",
    "vivo.com",
    "xiaomi.com",
}
RETAILER_DOMAINS = {
    "amazon.com",
    "amazon.co.uk",
    "amazon.de",
    "bestbuy.com",
    "jd.com",
    "sephora.com",
    "suning.com",
    "taobao.com",
    "tmall.com",
    "walmart.com",
}
REVIEW_DOMAINS = {
    "dpreview.com",
    "fstoppers.com",
    "incidecoder.com",
    "rtings.com",
}
COMMUNITY_DOMAINS = {
    "reddit.com",
    "weibo.com",
    "xiaohongshu.com",
    "zhihu.com",
}
SOURCE_PRIORITY = {
    "official": 1,
    "retailer": 2,
    "review": 3,
    "community": 4,
    "unknown": 5,
}


def clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def domain_matches(host: str, domain: str) -> bool:
    host = host.lower().removeprefix("www.")
    domain = domain.lower().removeprefix("www.")
    return host == domain or host.endswith(f".{domain}")


def classify_domain(host: str) -> str:
    if any(domain_matches(host, domain) for domain in OFFICIAL_DOMAINS):
        return "official"
    if any(domain_matches(host, domain) for domain in RETAILER_DOMAINS):
        return "retailer"
    if any(domain_matches(host, domain) for domain in REVIEW_DOMAINS):
        return "review"
    if any(domain_matches(host, domain) for domain in COMMUNITY_DOMAINS):
        return "community"
    return "unknown"


def canonicalize_url(value: str) -> tuple[str, str] | None:
    raw_url = html.unescape(value).strip().strip("<>'\"").rstrip(URL_TRAILING_CHARS)
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        return None

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username or parsed.password:
        return None

    host = parsed.hostname.lower().rstrip(".")
    port = parsed.port
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        host = f"{host}:{port}"

    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in TRACKING_QUERY_KEYS
    ]
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            host,
            parsed.path or "/",
            urlencode(sorted(query)),
            "",
        )
    )
    return normalized, parsed.hostname.lower().removeprefix("www.")


def extract_and_strip_urls(text: str) -> tuple[str, list[str], list[str]]:
    urls: list[str] = []
    domains: list[str] = []
    for match in URL_RE.findall(text):
        normalized = canonicalize_url(match)
        if normalized is None:
            continue
        url, domain = normalized
        if url not in urls:
            urls.append(url)
        if domain not in domains:
            domains.append(domain)

    stripped = URL_RE.sub(" ", text)
    stripped = re.sub(
        r"(?im)^\s*(?:参考来源|来源|参考链接|链接|网址)\s*[:：]?\s*$",
        "",
        stripped,
    )
    stripped = re.sub(r"(?i)(?:参考来源|来源|参考链接|链接|网址)\s*[:：]\s*(?=[，。；\n]|$)", "", stripped)
    return clean_text(stripped), urls, domains


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def reference_urls(annotation: dict[str, Any]) -> list[str]:
    urls = annotation.get("reference", {}).get("urls", [])
    normalized: list[str] = []
    for value in urls:
        result = canonicalize_url(str(value))
        if result is None:
            continue
        url, _ = result
        if url not in normalized:
            normalized.append(url)
    return normalized


def convert_tasks(
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    documents: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    seen_documents: set[tuple[str, str, str, tuple[str, ...]]] = set()
    duplicate_documents = 0
    total_urls_in_reasons = 0
    missing_source_documents = 0

    for task in tasks:
        task_id = clean_text(task.get("uuid"))
        question_text = clean_text(task.get("question"))
        products = task.get("product_list", [])
        if not task_id or not question_text or not products:
            raise ValueError(f"Invalid ShoppingComp task: {task_id or 'missing-id'}")

        product_document_ids: dict[str, list[str]] = {}
        scenes: set[str] = set()

        for product_index, product in enumerate(products):
            product_name = clean_text(product.get("product_name"))
            if not product_name:
                raise ValueError(f"Task {task_id} contains a product without a name")
            product_document_ids.setdefault(product_name, [])

            for annotation in product.get("scene_annotation_list", []):
                scene_id = clean_text(annotation.get("uuid"))
                raw_reason = clean_text(annotation.get("reason"))
                reason, inline_urls, inline_domains = extract_and_strip_urls(
                    raw_reason
                )
                total_urls_in_reasons += len(inline_urls)
                if not scene_id or not reason:
                    continue

                urls = list(
                    dict.fromkeys([*inline_urls, *reference_urls(annotation)])
                )
                url_domains = {
                    inline_domain
                    for inline_domain in inline_domains
                    if inline_domain
                }
                for url in urls:
                    normalized = canonicalize_url(url)
                    if normalized is not None:
                        url_domains.add(normalized[1])

                source_candidates = [
                    (classify_domain(domain), domain, url)
                    for url in urls
                    for normalized in [canonicalize_url(url)]
                    if normalized is not None
                    for domain in [normalized[1]]
                ]
                source_candidates.sort(
                    key=lambda item: (SOURCE_PRIORITY[item[0]], item[1], item[2])
                )
                if source_candidates:
                    source_type, source_domain, source_url = source_candidates[0]
                    source_status = "provided"
                else:
                    source_type, source_domain, source_url = (
                        "unknown",
                        "unknown",
                        "",
                    )
                    source_status = "missing"
                    missing_source_documents += 1

                dedupe_key = (product_name, reason, scene_id, tuple(urls))
                if dedupe_key in seen_documents:
                    duplicate_documents += 1
                    continue
                seen_documents.add(dedupe_key)

                document_id = f"shoppingcomp:{task_id}:{product_index}"
                chunk_id = f"{document_id}:{scene_id}"
                text = f"商品：{product_name}\n\n商品证据：{reason}"

                documents.append(
                    {
                        "id": chunk_id,
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "title": product_name,
                        "text": text,
                        "product_name": product_name,
                        "task_id": task_id,
                        "scene_id": scene_id,
                        "document_type": "product_evidence",
                        "source_urls": urls,
                        "source_url": source_url,
                        "source_domain": source_domain,
                        "source_domains": sorted(url_domains),
                        "source_type": source_type,
                        "source_status": source_status,
                        "dataset": "ShoppingComp",
                        "dataset_url": SOURCE_URL,
                        "language": "zh",
                        "license": LICENSE,
                    }
                )
                product_document_ids[product_name].append(document_id)
                scenes.add(scene_id)

        expected_document_ids = sorted(
            {
                document["document_id"]
                for document in documents
                if document["task_id"] == task_id
            }
        )
        expected_chunk_ids = sorted(
            {
                document["chunk_id"]
                for document in documents
                if document["task_id"] == task_id
            }
        )
        if not expected_document_ids or not expected_chunk_ids:
            raise ValueError(f"Task {task_id} has no product evidence documents")

        questions.append(
            {
                "question_id": task_id,
                "question": question_text,
                "expected_product_names": sorted(product_document_ids),
                "expected_document_ids": expected_document_ids,
                "expected_chunk_ids": expected_chunk_ids,
                "expected_scene_ids": sorted(scenes),
                "source_type": clean_text(task.get("source")),
                "task_type": "ecommerce_product_selection",
                "answerable": True,
                "dataset": "ShoppingComp",
                "language": "zh",
                "license": LICENSE,
            }
        )

    return (
        documents,
        questions,
        {
            "duplicate_documents_skipped": duplicate_documents,
            "documents_with_sources": len(documents) - missing_source_documents,
            "documents_missing_sources": missing_source_documents,
            "urls_extracted_from_reason_text": total_urls_in_reasons,
        },
    )


def convert_safety_tasks(
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        task_id = clean_text(task.get("uuid"))
        question_text = clean_text(task.get("question"))
        trap_rubric = clean_text(task.get("trap_rubric"))
        if not task_id or not question_text or not trap_rubric:
            raise ValueError(f"Invalid safety task: {task_id or 'missing-id'}")
        records.append(
            {
                "question_id": task_id,
                "question": question_text,
                "trap_rubric": trap_rubric,
                "task_type": "ecommerce_safety_evaluation",
                "expected_behavior": "识别风险或不当使用，并给出安全建议",
                "dataset": "ShoppingComp",
                "language": "zh",
                "license": LICENSE,
            }
        )
    return records


def evenly_spaced_sample(
    records: list[dict[str, Any]], size: int
) -> list[dict[str, Any]]:
    sample_size = max(1, min(size, len(records)))
    step = len(records) / sample_size
    return [records[int(index * step)] for index in range(sample_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    tasks = read_jsonl(args.input_dir / GT_FILE)
    safety_tasks = read_jsonl(args.input_dir / SAFETY_FILE)
    documents, questions, stats = convert_tasks(tasks)
    safety_questions = convert_safety_tasks(safety_tasks)
    sample = evenly_spaced_sample(questions, args.sample_size)

    write_jsonl(args.output_dir / "documents.jsonl", documents)
    write_jsonl(args.output_dir / "questions.jsonl", questions)
    write_jsonl(args.output_dir / "questions_sample.jsonl", sample)
    write_jsonl(args.output_dir / "safety_questions.jsonl", safety_questions)

    source_types = Counter(question["source_type"] for question in questions)
    evidence_source_types = Counter(
        document["source_type"] for document in documents
    )
    manifest = {
        "dataset": "ShoppingComp",
        "language": "zh",
        "source_url": SOURCE_URL,
        "paper": "https://www.arxiv.org/abs/2511.22978",
        "license": LICENSE,
        "modifications": (
            "Split product evidence into retrieval documents, normalized text, "
            "removed URLs from embedding/BM25 text, preserved canonical URLs and "
            "source-quality metadata, added stable RAG identifiers, expected "
            "product/document/chunk references, and separated safety tasks."
        ),
        "task_count": len(questions),
        "safety_task_count": len(safety_questions),
        "document_count": len(documents),
        "unique_product_count": len(
            {document["product_name"] for document in documents}
        ),
        "sample_question_count": len(sample),
        "source_types": dict(source_types),
        "evidence_source_types": dict(evidence_source_types),
        **stats,
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Documents: {len(documents)}")
    print(f"Questions: {len(questions)}")
    print(f"Safety questions: {len(safety_questions)}")
    print(f"Sample questions: {len(sample)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
