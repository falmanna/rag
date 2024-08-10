import logging

from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
from langchain_core.vectorstores import VectorStore
from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore
from langchain_elasticsearch.client import create_elasticsearch_client

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
            neo4j_log = logging.getLogger("neo4j")
            neo4j_log.setLevel(logging.CRITICAL)
            return Neo4jVector(
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                url=NEO4J_URL,
                embedding=get_embedding(device=embedding_device),
                search_type="hybrid",
                logger=neo4j_log,
            )
        case "elasticsearch":
            es_client = create_elasticsearch_client(url="http://localhost:9200")
            # if not es_client.indices.exists(index="elasticsearch-rag"):
            #     es_client.indices.create(
            #         index="elasticsearch-rag",
            #         body={
            #             "settings": {
            #                 "analysis": {
            #                     "analyzer": {
            #                         "arabic_analyzer": {
            #                             "tokenizer": "standard",
            #                             "filter": [
            #                                 "lowercase",
            #                                 "arabic_normalization",
            #                                 "arabic_stop",
            #                             ],
            #                         },
            #                         "english_analyzer": {
            #                             "tokenizer": "standard",
            #                             "filter": ["lowercase", "english_stop"],
            #                         },
            #                     },
            #                     "filter": {
            #                         "arabic_stop": {
            #                             "type": "stop",
            #                             "stopwords": "_arabic_",
            #                         },
            #                         "english_stop": {
            #                             "type": "stop",
            #                             "stopwords": "_english_",
            #                         },
            #                     },
            #                 }
            #             },
            #             "mappings": {
            #                 "properties": {
            #                     "content": {
            #                         "type": "text",
            #                         "fields": {
            #                             "arabic": {
            #                                 "type": "text",
            #                                 "analyzer": "arabic_analyzer",
            #                             },
            #                             "english": {
            #                                 "type": "text",
            #                                 "analyzer": "english_analyzer",
            #                             },
            #                         },
            #                     },
            #                     "vector": {
            #                         "type": "dense_vector",
            #                         "dims": EMVEDDING_DIMENSION,
            #                     },
            #                 }
            #             },
            #         },
            #     )
            return ElasticsearchStore(
                index_name="elasticsearch-rag",
                es_connection=es_client,
                embedding=get_embedding(device=embedding_device),
                strategy=DenseVectorStrategy(hybrid=True, text_field="text", rrf=False),
            )

        case _:
            raise ValueError(f"Unknown DB provider: {store}")
