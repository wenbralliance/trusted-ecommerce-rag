import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pymilvus import AsyncMilvusClient
from langchain_deepseek import ChatDeepSeek

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
milvus_client = AsyncMilvusClient(uri=MILVUS_URI)

embed_client = AsyncOpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url=os.getenv("EMBEDDING_BASE_URL"),
)

# 换成 DeepSeek
llm = ChatDeepSeek(
    model=os.getenv("LLM_MODEL", "deepseek-flash"),
    api_key=os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "False").lower() == "true"
TOP_N = int(os.getenv("TOP_N", "3"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_PRODUCTS = int(os.getenv("RERANK_TOP_PRODUCTS", "5"))
RERANK_EVIDENCE_PER_PRODUCT = int(
    os.getenv("RERANK_EVIDENCE_PER_PRODUCT", "1")
)
RERANK_DOC_MAX_CHARS = int(os.getenv("RERANK_DOC_MAX_CHARS", "4000"))


def _rerank_api_url() -> str:
    explicit_url = os.getenv("RERANK_API_URL", "").strip()
    if explicit_url:
        return explicit_url

    base_url = os.getenv("RERANK_BASE_URL", "").strip().rstrip("/")
    if base_url:
        if "/rerank/" in base_url or base_url.endswith("/rerank"):
            return base_url
        for suffix in ("/compatible-mode/v1", "/compatible-mode", "/v1"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        return (
            f"{base_url}/api/v1/services/"
            "rerank/text-rerank/text-rerank"
        )

    return (
        "https://dashscope.aliyuncs.com/api/v1/services/"
        "rerank/text-rerank/text-rerank"
    )


RERANK_API_URL = _rerank_api_url()
QUERY_MAX_CHARS = int(os.getenv("QUERY_MAX_CHARS", "4000"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
