from langchain_core.vectorstores import VectorStoreRetriever

from agent.utils.vectorstore import get_vectorstore


def get_vectorstore_retriever(k: int = 5) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_kwargs={"k": k})
