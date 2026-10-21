"""Ask one question through the complete RAG API and print a readable trace."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx
from fastapi.testclient import TestClient

from rag.main import app


FILTER_KEYS = {
    "source_type",
    "source_domain",
    "source_status",
    "task_id",
    "document_type",
    "dataset",
    "language",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="?",
        help="要提问的内容；省略后进入交互输入",
    )
    parser.add_argument(
        "--url",
        default="",
        help="已运行的服务地址，例如 http://localhost:8000",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "元数据过滤，可重复使用。支持 source_type、source_domain、"
            "source_status、task_id、document_type、dataset、language"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="HTTP 模式请求超时秒数",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出原始 JSON",
    )
    parser.add_argument(
        "--full-evidence",
        action="store_true",
        help="显示完整引用证据，而不是只显示前 500 字符",
    )
    return parser.parse_args()


def parse_filters(items: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError(f"Invalid filter {item!r}; expected KEY=VALUE")
        if key not in FILTER_KEYS:
            raise ValueError(
                f"Unsupported filter {key!r}; available: {sorted(FILTER_KEYS)}"
            )
        filters[key] = value
    return filters


def request_question(
    payload: dict[str, Any],
    *,
    url: str,
    timeout: float,
) -> dict[str, Any]:
    if url:
        response = httpx.post(
            f"{url.rstrip('/')}/api/v1/chat",
            json=payload,
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:1000]}"
            )
        return response.json()

    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:1000]}"
            )
        return response.json()


def print_result(data: dict[str, Any], full_evidence: bool) -> None:
    print("\n" + "=" * 80)
    print(f"请求 ID: {data.get('request_id', '')}")
    print(f"检索证据数: {data.get('retrieval_count', 0)}")
    print(f"是否降级: {data.get('is_degraded', False)}")
    print("\n回答:")
    print(data.get("answer", ""))

    verified = data.get("verified_citations", [])
    print(f"\n已验证引用: {len(verified)}")
    for index, citation in enumerate(verified, start=1):
        print(f"\n[{index}] {citation.get('title') or '未知标题'}")
        print(f"    document_id: {citation.get('document_id', '')}")
        print(f"    chunk_id:    {citation.get('chunk_id', '')}")
        print(
            "    来源: "
            f"{citation.get('source_domain') or 'unknown'} / "
            f"{citation.get('source_type') or 'unknown'} / "
            f"{citation.get('source_status') or 'unknown'}"
        )
        source_url = citation.get("source_url")
        if source_url:
            print(f"    URL: {source_url}")
        evidence = citation.get("evidence") or ""
        if evidence:
            shown = evidence if full_evidence else evidence[:500]
            print(f"    证据: {shown}")
            if not full_evidence and len(evidence) > len(shown):
                print("    ...（使用 --full-evidence 显示完整证据）")

    unverified = data.get("unverified_citations", [])
    print(f"\n未验证引用: {len(unverified)}")
    for index, citation in enumerate(unverified, start=1):
        print(
            f"[{index}] document_id={citation.get('document_id', '')} "
            f"chunk_id={citation.get('chunk_id', '')}"
        )

    alerts = data.get("alerts", [])
    print(f"\n告警: {len(alerts)}")
    for index, alert in enumerate(alerts, start=1):
        print(f"[{index}] {alert}")
    print("=" * 80)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    question = (args.question or input("请输入问题: ")).strip()
    if not question:
        raise SystemExit("问题不能为空")

    payload: dict[str, Any] = {"query": question}
    filters = parse_filters(args.filter)
    if filters:
        payload["filters"] = filters

    try:
        result = request_question(
            payload,
            url=args.url,
            timeout=args.timeout,
        )
    except Exception as exc:
        raise SystemExit(f"请求失败: {exc}") from None

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result, full_evidence=args.full_evidence)


if __name__ == "__main__":
    main()
