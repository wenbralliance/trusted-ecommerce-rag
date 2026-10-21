import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from rag.config import COLLECTION_NAME

# 注意：这里的路径是从项目根目录运行的，所以可以直接找到 .env
load_dotenv()

client = MilvusClient(uri=os.getenv("MILVUS_URI"))

embed_client = OpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url=os.getenv("EMBEDDING_BASE_URL"),
)


def embed_query(text: str) -> list[float]:
    resp = embed_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=[text],
    )
    return resp.data[0].embedding


def hybrid_search(query: str, top_k: int = 3):
    query_dense = embed_query(query)

    dense_req = AnnSearchRequest(
        data=[query_dense], anns_field="dense",
        param={"metric_type": "IP"}, limit=10
    )
    sparse_req = AnnSearchRequest(
        data=[query], anns_field="sparse",
        param={"metric_type": "BM25"}, limit=10
    )

    return client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=["text", "chunk_id", "document_id", "title"],
    )


def test_ecommerce_product_search():
    """测试电商商品问题能否召回带元数据的商品证据"""
    query = "索尼 α6700 支持4K视频和动物眼部对焦吗？"
    results = hybrid_search(query)

    titles = []
    for hits in results:
        for hit in hits:
            titles.append(hit.entity.get("title"))

    assert titles, "商品检索没有返回结果"
    assert any("索尼" in str(title) for title in titles), (
        f"预期召回索尼商品证据，实际标题为 {titles}"
    )
    print(f"\n成功召回商品：{titles}")
