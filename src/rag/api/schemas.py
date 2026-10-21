from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.config import QUERY_MAX_CHARS


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal[
        "official", "retailer", "review", "community", "unknown"
    ] | None = None
    source_domain: str | None = Field(default=None, max_length=256)
    source_status: Literal["provided", "missing"] | None = None
    task_id: str | None = Field(default=None, max_length=64)
    document_type: str | None = Field(default=None, max_length=64)
    dataset: str | None = Field(default=None, max_length=64)
    language: str | None = Field(default=None, max_length=16)


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=QUERY_MAX_CHARS,
        description="用户提问的内容",
        examples=["我想买一台适合旅行拍摄的人像相机"],
    )
    filters: SearchFilters | None = Field(
        default=None,
        description="可选的元数据过滤条件",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class CitationItem(BaseModel):
    document_id: str = Field(description="检索结果中的 document_id")
    chunk_id: str = Field(description="检索结果中的 chunk_id")
    title: str | None = Field(default=None, description="商品或文档标题")
    source_url: str | None = Field(default=None, description="首选来源地址")
    source_urls: list[str] = Field(
        default_factory=list,
        description="规范化后的全部来源地址",
    )
    source_domain: str | None = Field(default=None, description="首选来源域名")
    source_type: str | None = Field(
        default=None,
        description="official、retailer、review、community 或 unknown",
    )
    source_status: str | None = Field(
        default=None,
        description="provided、missing 或 unverified",
    )
    evidence: str | None = Field(
        default=None,
        description="用于支持答案的检索原文",
    )


class ChatResponse(BaseModel):
    request_id: str = Field(description="用于日志排查的请求 ID")
    answer: str = Field(description="大模型生成的回答内容")
    verified_citations: list[CitationItem] = Field(
        default_factory=list,
        description="文档 ID 和块 ID 均通过后验校验的引用",
    )
    unverified_citations: list[CitationItem] = Field(
        default_factory=list,
        description="未出现在本次检索结果中的引用",
    )
    retrieval_count: int = Field(
        default=0,
        description="本次请求最终保留的去重检索结果数量",
    )
    is_degraded: bool = Field(
        default=False,
        description="是否发生检索或重排降级",
    )
    alerts: list[str] = Field(
        default_factory=list,
        description="降级、来源缺失或其他需要关注的信息",
    )


class HealthResponse(BaseModel):
    status: str = Field(description="服务与集合就绪状态")
    collection: str = Field(description="当前 Milvus 集合名")
    row_count: int = Field(description="集合中的记录数")
