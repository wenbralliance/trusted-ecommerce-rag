import asyncio
from rag.config import milvus_client, COLLECTION_NAME
from rag.db.schema import OUTPUT_FIELDS


async def check_data():
    print(f"--- 正在查询表: {COLLECTION_NAME} ---")

    # 必须加 await
    stats = await milvus_client.get_collection_stats(collection_name=COLLECTION_NAME)
    print(f"总数据量: {stats['row_count']} 条")

    # 必须加 await
    results = await milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter="chunk_id != ''",
        output_fields=OUTPUT_FIELDS,
    )

    for i, row in enumerate(results):
        print(f"\n[{i + 1}] 文档: {row['document_id']} | ID: {row['chunk_id']}")
        print(f"标题: {row.get('title', '')} | 来源: {row.get('source_url', '')}")
        print(f"内容: {row['text']}")


if __name__ == "__main__":
    # 必须用 asyncio.run 启动
    asyncio.run(check_data())
