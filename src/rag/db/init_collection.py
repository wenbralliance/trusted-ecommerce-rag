import argparse
import asyncio

from pymilvus import DataType, Function, FunctionType

from rag.config import COLLECTION_NAME, EMBEDDING_DIM, milvus_client


async def init_db(force_recreate: bool = False):
    """
    初始化 Milvus 数据库表
    :param force_recreate: 如果为 True，且表存在，会先删除旧表再重建（慎用！）
    """
    # 检查表是否已存在
    has_collection = await milvus_client.has_collection(collection_name=COLLECTION_NAME)

    if has_collection:
        if force_recreate:
            print(f"[WARN] Dropping existing collection: {COLLECTION_NAME}")
            await milvus_client.drop_collection(collection_name=COLLECTION_NAME)
        else:
            print(f"[INFO] Collection already exists: {COLLECTION_NAME}")
            return

    print(f"[INFO] Creating schema for collection: {COLLECTION_NAME}")

    # 1. 定义 Schema
    schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=False)

    # 主键和基本文本
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192,
                     enable_analyzer=True)  # 开启分词，供 BM25 使用

    # 向量字段
    schema.add_field(
        field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM
    )
    schema.add_field(
        field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR
    )

    # 元数据字段
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="source_url", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(
        field_name="source_domain", datatype=DataType.VARCHAR, max_length=256
    )
    schema.add_field(
        field_name="source_type", datatype=DataType.VARCHAR, max_length=32
    )
    schema.add_field(field_name="source_urls", datatype=DataType.JSON)
    schema.add_field(
        field_name="source_status", datatype=DataType.VARCHAR, max_length=32
    )
    schema.add_field(field_name="task_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(
        field_name="document_type", datatype=DataType.VARCHAR, max_length=64
    )
    schema.add_field(field_name="dataset", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=16)

    # 2. 增加 BM25 函数（自动把 text 转为 sparse 向量）
    schema.add_function(
        Function(
            name="bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse"],
        )
    )

    # 3. 准备索引参数
    index_params = milvus_client.prepare_index_params()
    # 稠密索引 (HNSW 或 AUTOINDEX)
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="IP")
    # 稀疏索引 (BM25)
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

    # 4. 创建表
    await milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"[INFO] Collection created: {COLLECTION_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the RAG collection")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the existing collection before creating it again",
    )
    args = parser.parse_args()
    asyncio.run(init_db(force_recreate=args.recreate))
