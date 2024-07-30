from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
from langchain_core.vectorstores import VectorStore

from agent.utils.embedding import get_embedding
from configs import (
    EMVEDDING_DIMENSION,
    PGVECTOR_DATABASE,
    PGVECTOR_PASSWORD,
    PGVECTOR_PORT,
    PGVECTOR_USER,
    VECTOR_STORE,
)


def get_vectorstore(
    *, store: str = VECTOR_STORE, embedding_device: str = None
) -> VectorStore:
    match store:
        case "pgvector-rs":
            return PGVecto_rs(
                embedding=get_embedding(device=embedding_device),
                collection_name="rag-pgvector-rs",
                db_url=f"postgresql+psycopg://{PGVECTOR_USER}:{PGVECTOR_PASSWORD}@localhost:{PGVECTOR_PORT}/{PGVECTOR_DATABASE}",
                dimension=EMVEDDING_DIMENSION,
            )

        case _:
            raise ValueError(f"Unknown DB provider: {store}")
