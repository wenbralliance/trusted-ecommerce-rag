from langgraph.graph import StateGraph, START, END

from rag.graph.state import AgentState
from rag.graph.nodes import (
    agent_node,
    generate_node,
    grade_documents_node,
    retrieve_node,
    rewrite_node,
)
from rag.config import MAX_RETRIES

# 2. 初始化图
workflow = StateGraph(AgentState)

# 3. 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)

# 4. 连线
workflow.add_edge(START, "agent")


# 条件边 1：Agent 之后的判定
def should_continue(state: AgentState):
    # 达到上限后进入 generate，由节点返回结构化的拒答结果。
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "generate"

    last_message = state["messages"][-1]
    # 如果大模型决定调用工具，就去检索
    if last_message.tool_calls:
        return "retrieve"
    # 如果大模型没调用工具（直接回答了），就结束
    return END


workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"retrieve": "retrieve", "generate": "generate", END: END},
)

# 条件边 2：Retrieve 之后的判定 (Grader 打分)
# grade_documents_node 内部直接返回 "generate" 或 "rewrite"
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_node,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

# 普通边：Rewrite 之后回到 Agent 重新决策
workflow.add_edge("rewrite", "agent")

# 普通边：Generate 之后直接结束
workflow.add_edge("generate", END)

# 5. 编译图
app = workflow.compile()
