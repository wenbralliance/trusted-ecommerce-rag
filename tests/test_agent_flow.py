import pytest
from langchain_core.messages import HumanMessage
from rag.graph.builder import app


# 标记这个测试为异步测试
@pytest.mark.asyncio
async def test_agent_end_to_end():
    print("\n=== 启动 Agentic RAG E2E 测试 ===")
    query = "我想买一台适合旅行拍摄、支持4K视频和动物眼部对焦的相机"
    print(f"用户提问: {query}\n")

    inputs = {"messages": [HumanMessage(content=query)]}

    # 运行图
    result = await app.ainvoke(inputs)

    # 获取最后一条消息
    final_msg = result["messages"][-1]

    # 验证是否输出了正确的结果
    assert final_msg.content is not None, "最终生成的回答不能为空"
    print("\n=== 最终回答 ===")
    print(final_msg.content)

    # 你可以继续断言内容里是否包含预期的关键词
    # assert "买家" in final_msg.content
