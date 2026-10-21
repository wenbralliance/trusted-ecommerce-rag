from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routes import router
from rag.config import CORS_ORIGINS

app = FastAPI(
    title="Trusted Agentic RAG API",
    description="基于 LangGraph、Milvus 和可审计来源元数据的电商 RAG 服务",
    version="1.1.0",
)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=CORS_ORIGINS != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(router, prefix="/api/v1", tags=["Agentic Chat"])
