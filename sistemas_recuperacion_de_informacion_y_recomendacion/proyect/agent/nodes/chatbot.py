import json

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.prompts.prompts import rag_prompt
from agent.schemas.states import ChatbotState
from agent.tools.rag import rag_tool
from core.config import MODEL

llm = ChatOpenAI(model=MODEL, temperature=0.3)
llm_with_tool = llm.bind_tools([rag_tool])


def chatbot_node(state: ChatbotState) -> ChatbotState:
    """Processes user messages with tool support."""
    user_request = state["user_input"].content
    messages = [
        SystemMessage(
            content=f"You are a virtual assistant. This is the chat history: {state['messages'][-5:]}"
        ),
        user_request,
    ]
    last_message = SystemMessage(content="")
    if state["messages"]:
        last_message = state["messages"][-1]

    if last_message and isinstance(last_message, ToolMessage):
        tool_response = last_message.content
        rag_response = json.loads(tool_response)
        response = llm.invoke(
            [
                SystemMessage(
                    content=rag_prompt(
                        rag_response.get("context"),
                        user_request,
                        state["messages"][-5:],
                    )
                )
            ]
        )
        state["ai_response"] = response
    elif last_message.additional_kwargs and last_message.tool_calls:
        state["messages"].append(last_message)
    else:
        response = llm_with_tool.invoke(messages)
        state["ai_response"] = response

    state["messages"].append(response)

    return state
