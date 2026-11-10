"""Evidence-grounded answer generation and citation derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from rag.prompt.prompt_template import grounded_generate_prompt


class EvidenceClaim(BaseModel):
    text: str = Field(description="回答中使用的一条可独立核验的原子事实")
    evidence_ids: list[str] = Field(
        description="直接支持该事实的证据编号，例如 E1"
    )


class GroundedAnswer(BaseModel):
    answer: str = Field(description="由受支持事实组成的最终回答")
    claims: list[EvidenceClaim] = Field(
        default_factory=list,
        description="回答所依据的、带证据编号的原子事实",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="现有证据无法确认的用户关键条件",
    )


@dataclass(frozen=True)
class GroundingSummary:
    claim_count: int
    cited_claim_count: int
    citation_validity_rate: float | None
    missing_information_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_count": self.claim_count,
            "cited_claim_count": self.cited_claim_count,
            "citation_validity_rate": self.citation_validity_rate,
            "missing_information_count": self.missing_information_count,
        }


def format_numbered_evidence(
    evidence: list[tuple[str, str]],
) -> tuple[str, dict[str, int]]:
    """Format evidence as E1..En and return the label-to-index mapping."""
    blocks: list[str] = []
    label_to_index: dict[str, int] = {}
    for index, (title, text) in enumerate(evidence, start=1):
        label = f"E{index}"
        label_to_index[label] = index - 1
        blocks.append(
            "\n".join(
                [
                    f"[{label}]",
                    f"标题：{title or '未知文档'}",
                    "正文：",
                    text.strip(),
                ]
            )
        )
    return "\n\n".join(blocks), label_to_index


def format_source_evidence(
    sources: list[dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    return format_numbered_evidence(
        [
            (
                str(source.get("title") or "未知文档"),
                str(source.get("text") or ""),
            )
            for source in sources
        ]
    )


def build_grounded_prompt(question: str, numbered_evidence: str) -> str:
    return (
        f"{grounded_generate_prompt}\n\n"
        f"用户问题：\n{question.strip()}\n\n"
        f"检索证据：\n{numbered_evidence.strip()}"
    )


def normalize_evidence_ids(values: list[str], count: int) -> list[int]:
    """Return unique valid zero-based indexes from model-generated E labels."""
    indexes: list[int] = []
    for value in values:
        raw = str(value).strip().upper()
        if raw.startswith("E"):
            raw = raw[1:]
        try:
            index = int(raw) - 1
        except ValueError:
            continue
        if 0 <= index < count and index not in indexes:
            indexes.append(index)
    return indexes


def derive_citation_indexes(
    claims: list[EvidenceClaim],
    source_count: int,
) -> list[int]:
    indexes: list[int] = []
    for claim in claims:
        for index in normalize_evidence_ids(claim.evidence_ids, source_count):
            if index not in indexes:
                indexes.append(index)
    return indexes


def summarize_grounding(
    claims: list[EvidenceClaim],
    source_count: int,
    missing_information: list[str],
) -> GroundingSummary:
    cited = sum(
        bool(normalize_evidence_ids(claim.evidence_ids, source_count))
        for claim in claims
    )
    validity_rate = cited / len(claims) if claims else None
    return GroundingSummary(
        claim_count=len(claims),
        cited_claim_count=cited,
        citation_validity_rate=validity_rate,
        missing_information_count=len(
            [item for item in missing_information if item.strip()]
        ),
    )


async def generate_grounded_answer(
    question: str,
    sources: list[dict[str, Any]],
    generator: Callable[[str], Awaitable[GroundedAnswer]],
) -> tuple[GroundedAnswer, list[int], GroundingSummary]:
    evidence, _ = format_source_evidence(sources)
    response = await generator(build_grounded_prompt(question, evidence))
    citation_indexes = derive_citation_indexes(response.claims, len(sources))
    summary = summarize_grounding(
        response.claims,
        len(sources),
        response.missing_information,
    )
    return response, citation_indexes, summary
