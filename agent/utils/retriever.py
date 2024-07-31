from langchain.retrievers import ContextualCompressionRetriever

from agent.utils.rerank import get_reranker
from agent.utils.vectorstore import get_vectorstore


def get_vectorstore_retriever(*, limit: int = 5, embedding_device: str = None):
    return get_vectorstore(embedding_device=embedding_device).as_retriever(
        search_kwargs={"k": limit}
    )


def get_retriever(*, rerank: bool = True, limit: int = 5, embedding_device: str = None):
    if rerank:
        return ContextualCompressionRetriever(
            base_compressor=get_reranker(
                embedding_device=embedding_device, limit=limit
            ),
            base_retriever=get_vectorstore_retriever(
                limit=limit * 4, embedding_device=embedding_device
            ),
        )
    else:
        return get_vectorstore_retriever(limit=limit, embedding_device=embedding_device)
