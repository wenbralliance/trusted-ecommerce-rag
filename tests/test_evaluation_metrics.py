from evaluate_retrieval import _score_products


def test_retrieval_metrics_reward_relevant_products_at_higher_ranks():
    expected = {"Product A", "Product B"}
    strong = _score_products(
        expected,
        [{"title": "Product A"}, {"title": "Product B"}],
    )
    weak = _score_products(
        expected,
        [
            {"title": "Product X"},
            {"title": "Product Y"},
            {"title": "Product A"},
            {"title": "Product B"},
        ],
    )

    assert strong["recall"] == 1.0
    assert weak["recall"] == 1.0
    assert strong["ndcg_at_5"] == 1.0
    assert weak["ndcg_at_5"] < strong["ndcg_at_5"]
    assert weak["average_precision"] < strong["average_precision"]
    assert strong["reciprocal_rank"] == 1.0
    assert weak["reciprocal_rank"] == 1 / 3
