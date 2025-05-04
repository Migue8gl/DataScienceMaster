import json
import logging
from typing import Tuple

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sentence_transformers import SentenceTransformer

from agent.nodes.chatbot import chatbot_node
from agent.schemas.states import ChatbotState
from agent.tools.rag import rag_tool
from components.chunker import Chunker
from components.retriever import Retriever
from components.vectordb import VectorDB
from strategies.chunker_strategies import SemanticChunker
from strategies.retriever_strategies import HybridRetriever
from utils.utils import read_files_from_directory

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ChatAgent:
    def __init__(self):
        self.graph = None
        self.state = None
        self.initialize_agent()

    def initialize_agent(self):
        """Initialize LangGraph graph and vector database"""
        logger.debug("Reading documents from ./data/docs...")
        documents = read_files_from_directory("./data/docs")
        logger.debug("Initializing embedding model...")
        model = SentenceTransformer("intfloat/multilingual-e5-base")

        logger.debug("Creating chunker, retriever, and vector DB...")
        chunker = Chunker(SemanticChunker(model))
        retriever = Retriever(HybridRetriever(model, 0.8))
        db = VectorDB(model, chunker, retriever)

        logger.debug("Creating index in the vector DB...")
        db.create_index(documents, override=False)

        logger.debug("Building LangGraph flow...")
        graph_builder = StateGraph(ChatbotState)
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_node("tool_node", ToolNode(tools=[rag_tool]))
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
            {"tools": "tool_node", END: END},
        )
        graph_builder.add_edge("tool_node", "chatbot")

        self.graph = graph_builder.compile()
        self.state = ChatbotState(
            user_input=HumanMessage(content=""),
            ai_response=AIMessage(content=""),
            messages=[],
            vector_db=db,
        )
        logger.debug("Agent initialized successfully.")

    def generate_response(self, prompt: str) -> Tuple[str, bool]:
        """Generate agent response for a given prompt"""
        logger.debug(f"STATE: {self.state}")
        self.state["user_input"] = HumanMessage(content=prompt)
        logger.debug(f"Received user input: {prompt}")

        full_response = ""
        sources = []
        used_rag = False

        for event in self.graph.stream(self.state):
            for key, value in event.items():
                if key == "tool_node":
                    used_rag = True
                    sources = json.loads(value.get("messages")[0].content)["source"]
                if "ai_response" in value and value["ai_response"].content:
                    full_response = value["ai_response"].content

        logger.debug(f"Full AI response: {full_response}")

        return full_response, {"used_rag": used_rag, "sources": sources}


def setup_streamlit_ui():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("🤖 Assistant with RAG")
    st.markdown("##### By Miguel García López")

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "agent" not in st.session_state:
        with st.spinner("Initializing agent..."):
            logger.debug("Initializing ChatAgent inside Streamlit session.")
            st.session_state.agent = ChatAgent()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "How can I assist you today?",
                "used_rag": False,
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("used_rag", False):
                st.success("🔍 This answer was enhanced using RAG capabilities")
                sources = msg.get("sources", [])
                if sources:
                    with st.expander("View sources"):
                        st.write("### Sources used:")
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. {source}")

    disabled = st.session_state.is_processing
    if prompt := st.chat_input("Type your message...", disabled=disabled):
        st.session_state.pending_prompt = prompt
        st.session_state.is_processing = True
        st.rerun()

    if st.session_state.pending_prompt and st.session_state.is_processing:
        prompt = st.session_state.pending_prompt

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    full_response, metadata = st.session_state.agent.generate_response(
                        prompt
                    )

                    response_placeholder.markdown(full_response)

                    if metadata.get("used_rag", False):
                        st.success("🔍 This answer was enhanced using RAG capabilities")
                        sources = metadata.get("sources", [])
                        if sources:
                            with st.expander("View sources"):
                                st.write("### Sources used:")
                                for i, source in enumerate(sources, 1):
                                    st.write(f"{i}. {source}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "used_rag": metadata.get("used_rag"),
                            "sources": metadata.get("sources", []),
                        }
                    )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Error generating response: {e}")
                finally:
                    st.session_state.pending_prompt = None
                    st.session_state.is_processing = False
                    st.rerun()


if __name__ == "__main__":
    setup_streamlit_ui()
