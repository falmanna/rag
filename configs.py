import os

from dotenv import load_dotenv

load_dotenv()

# Database configuration
VECTOR_STORE = os.environ.get("VECTOR_STORE")

NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")

PGVECTOR_HOST = os.environ.get("PGVECTOR_HOST", "pgvector")
PGVECTOR_PORT = int(os.environ.get("PGVECTOR_PORT", 5432))
PGVECTOR_USER = os.environ.get("PGVECTOR_USER", "postgres")
PGVECTOR_PASSWORD = os.environ.get("PGVECTOR_PASSWORD", "feras123456")
PGVECTOR_DATABASE = os.environ.get("PGVECTOR_DATABASE", "feras")

# Embeddings configuration
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE")
EMVEDDING_DIMENSION = int(os.environ.get("EMVEDDING_DIMENSION"))

# Reranking configs
RERANKING_PROVIDER = os.environ.get("RERANKING_PROVIDER")
RERANKING_MODEL_NAME = os.environ.get("RERANKING_MODEL_NAME")

# LLM configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME")

# Ingestion configuration
CHUNK_CHARACTER_SIZE = int(os.environ.get("CHUNK_CHARACTER_SIZE"))
CHUNK_CHARACTER_OVERLAP = int(os.environ.get("CHUNK_CHARACTER_OVERLAP"))
CHUNK_QUEUE_MAX_SIZE = int(os.environ.get("CHUNK_QUEUE_MAX_SIZE"))
CHUNK_INDEXING_BATCH_SIZE = int(os.environ.get("CHUNK_INDEXING_BATCH_SIZE"))
CHUNK_MIN_SIZE = int(os.environ.get("CHUNK_MIN_SIZE"))
NUMBER_OF_CORES = int(os.environ.get("NUMBER_OF_CORES", os.cpu_count()))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", NUMBER_OF_CORES))
