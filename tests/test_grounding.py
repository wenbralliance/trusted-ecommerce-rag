from rag.graph.grounding import (
    EvidenceClaim,
    derive_citation_indexes,
    normalize_evidence_ids,
    summarize_grounding,
)


def test_evidence_ids_are_normalized_and_deduplicated():
    assert normalize_evidence_ids(["E2", "2", "e1", "E9", "bad"], 3) == [
        1,
        0,
    ]


def test_claim_citations_are_derived_only_from_valid_evidence():
    claims = [
        EvidenceClaim(text="Fact one", evidence_ids=["E1", "E99"]),
        EvidenceClaim(text="Fact two", evidence_ids=["E2"]),
        EvidenceClaim(text="No evidence", evidence_ids=["missing"]),
    ]

    indexes = derive_citation_indexes(claims, source_count=2)
    summary = summarize_grounding(
        claims,
        source_count=2,
        missing_information=["Price", "Warranty"],
    )

    assert indexes == [0, 1]
    assert summary.claim_count == 3
    assert summary.cited_claim_count == 2
    assert summary.citation_validity_rate == 2 / 3
    assert summary.missing_information_count == 2


def test_grounding_rate_is_none_when_no_claims_are_returned():
    summary = summarize_grounding([], source_count=3, missing_information=[])
    assert summary.citation_validity_rate is None
