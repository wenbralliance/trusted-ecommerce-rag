"""Shared Milvus schema constants."""

OUTPUT_FIELDS = [
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

FILTERABLE_FIELDS = {
    "document_type",
    "source_domain",
    "source_status",
    "source_type",
    "task_id",
    "language",
    "dataset",
}

REQUIRED_INSERT_FIELDS = [
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
