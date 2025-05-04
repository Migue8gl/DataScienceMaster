from typing import List


def rag_prompt(context: str, user_request: str, chat_history: List[str]) -> str:
    return f"""You are a helpful AI assistant with access to a knowledge base through a RAG system.
    ### RETRIEVED CONTEXT
    The following information was retrieved based on the user's query:
    {context}

    ### CONVERSATION HISTORY
    {chat_history}

    ### INSTRUCTIONS
    1. Your primary goal is to answer the user's question accurately and completely.
    2. Format your response for readability using markdown when appropriate.
    3. If there is not enough data to answer (extreme case), tell the user.

    ### USER REQUEST
    {user_request}
    """
