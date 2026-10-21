"""Batch-ingest the prepared ShoppingComp JSONL corpus into Milvus."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, RateLimitError

from rag.config import (
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    embed_client,
    milvus_client,
)
from rag.db.init_collection import init_db


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "ecommerce"
    / "shoppingcomp"
    / "documents.jsonl"
)
URL_RE = re.compile(r"https?://", re.IGNORECASE)


def read_documents(path: Path, limit: int | None) -> list[dict[str, Any]]:
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
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def prepare_record(record: dict[str, Any]) -> dict[str, Any]:
    required = [
        "id",
        "text",
        "document_id",
        "chunk_id",
        "title",
        "source_url",
        "source_domain",
        "source_type",
        "source_urls",
        "source_status",
        "task_id",
        "document_type",
        "dataset",
        "language",
    ]
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            f"Document {record.get('id', 'unknown')} is missing fields: {missing}"
        )
    if not record["text"].strip() or URL_RE.search(record["text"]):
        raise ValueError(
            f"Document {record['id']} must contain clean text without URLs"
        )

    limits = {
        "id": 128,
        "text": 8192,
        "document_id": 256,
        "chunk_id": 128,
        "title": 512,
        "source_url": 2048,
        "source_domain": 256,
        "source_type": 32,
        "source_status": 32,
        "task_id": 64,
        "document_type": 64,
        "dataset": 64,
        "language": 16,
    }
    for field, maximum in limits.items():
        value = record[field]
        if not isinstance(value, str):
            raise ValueError(f"Document {record['id']} field {field} must be a string")
        if len(value) > maximum:
            raise ValueError(
                f"Document {record['id']} field {field} exceeds {maximum} characters"
            )

    source_urls = record["source_urls"]
    if not isinstance(source_urls, list) or not all(
        isinstance(url, str) for url in source_urls
    ):
        raise ValueError(
            f"Document {record['id']} source_urls must be a list of strings"
        )

    return {
        "id": record["id"],
        "text": record["text"],
        "document_id": record["document_id"],
        "chunk_id": record["chunk_id"],
        "title": record["title"],
        "source_url": record["source_url"],
        "source_domain": record["source_domain"],
        "source_type": record["source_type"],
        "source_urls": source_urls,
        "source_status": record["source_status"],
        "task_id": record["task_id"],
        "document_type": record["document_type"],
        "dataset": record["dataset"],
        "language": record["language"],
    }


async def embed_batch(texts: list[str]) -> list[list[float]]:
    for attempt in range(3):
        try:
            response = await embed_client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL"),
                input=texts,
            )
            break
        except (APIConnectionError, APITimeoutError, RateLimitError):
            if attempt == 2:
                raise
            await asyncio.sleep(2**attempt)

    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = [item.embedding for item in ordered]
    bad_dimensions = [len(vector) for vector in vectors if len(vector) != EMBEDDING_DIM]
    if bad_dimensions:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
            f"received {bad_dimensions[0]}"
        )
    return vectors


async def ingest(
    records: list[dict[str, Any]],
    batch_size: int,
    recreate: bool,
    dry_run: bool,
) -> None:
    if not records:
        raise ValueError("No documents to ingest")
    # The configured Qwen embedding endpoint accepts at most 25 texts per call.
    batch_size = max(1, min(batch_size, 25))
    records = [prepare_record(record) for record in records]
    print(f"Validated {len(records)} documents.")

    if dry_run:
        print("Dry run complete; Milvus and Embedding API were not called.")
        return

    await init_db(force_recreate=recreate)
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = await embed_batch([record["text"] for record in batch])
        payload = [
            {**record, "dense": vector}
            for record, vector in zip(batch, vectors, strict=True)
        ]
        result = await milvus_client.upsert(
            collection_name=COLLECTION_NAME,
            data=payload,
        )
        inserted = int(
            result.get(
                "upsert_count",
                result.get("insert_count", len(batch)),
            )
        )
        total += inserted
        print(f"Upserted {total}/{len(records)}")

    await milvus_client.flush(collection_name=COLLECTION_NAME)
    print(f"Ingestion complete: {total} documents in {COLLECTION_NAME}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingestion",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N documents",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate JSON and field limits without calling external services",
    )
    args = parser.parse_args()
    records = read_documents(args.input, args.limit)
    asyncio.run(
        ingest(
            records,
            batch_size=args.batch_size,
            recreate=args.recreate,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
