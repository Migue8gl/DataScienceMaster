from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage

from components.vectordb import VectorDB
from langgraph.graph.message import add_messages


class ChatbotState(TypedDict):
    """Represents the state of an user in the conversation flow."""

    messages: Annotated[list, add_messages]
    user_input: HumanMessage
    ai_response: AIMessage
    vector_db: VectorDB
