import operator
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retry_count: NotRequired[int]
    filters: NotRequired[dict[str, Any]]
    retrieved_sources: Annotated[list[dict[str, Any]], operator.add]
    retrieval_alerts: Annotated[list[str], operator.add]
