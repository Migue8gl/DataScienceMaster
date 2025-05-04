import logging
from typing import Annotated, Dict, List

from langchain_core.tools import tool

from components.vectordb import VectorDB
from langgraph.prebuilt import InjectedState
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@tool
def rag_tool(
    query: str, db: Annotated[VectorDB, InjectedState("vector_db")], k: int = 5
) -> List[Dict[str, str]]:
    """
    Use it when asking questions related to the technical field of informatics or machine learning or data science.
    
    Retrieves relevant context using a Retrieval-Augmented Generation (RAG) approach
    by querying a vector database with the input string. It will return a dictionary with the more
    relevant documents and their contents.

    Args:
        query (str): The input query to search for relevant context.
        db (str): Injected vector database used for similarity search.
        k (int, optional): The number of top results to retrieve. Defaults to 5.

    Returns:
        List[Dict[str, str]]: A list of the top-k retrieved context strings. Key is the document filename and value the content
    """
    results = db.search(query, k)

    return {
        "source": [filename for result in results for filename in result.keys()],
        "context": [context for result in results for context in result.values()],
    }
