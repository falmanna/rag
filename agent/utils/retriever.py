from langchain_core.vectorstores import VectorStoreRetriever

from agent.utils.vectorstore import get_vectorstore


def get_vectorstore_retriever(
    *, limit: int = 5, embedding_device: str = None
) -> VectorStoreRetriever:
    return get_vectorstore(embedding_device=embedding_device).as_retriever(
        search_kwargs={"k": limit}
    )
