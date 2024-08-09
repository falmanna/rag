from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
from langchain_core.vectorstores import VectorStore

from agent.utils.embedding import get_embedding
from configs import (
    EMVEDDING_DIMENSION,
    NEO4J_PASSWORD,
    NEO4J_URL,
    NEO4J_USERNAME,
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

        case "neo4j":
            return Neo4jVector(
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                url=NEO4J_URL,
                embedding=get_embedding(device=embedding_device),
                search_type="hybrid",
            )

        case _:
            raise ValueError(f"Unknown DB provider: {store}")
