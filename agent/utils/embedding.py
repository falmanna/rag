import os

from langchain_community.embeddings import InfinityEmbeddings, InfinityEmbeddingsLocal
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from configs import EMBEDDING_DEVICE, EMBEDDING_MODEL_NAME, EMBEDDING_PROVIDER


def get_embedding(*, device: str = None) -> Embeddings:
    match EMBEDDING_PROVIDER:
        case "huggingface":
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device or EMBEDDING_DEVICE},
                cache_folder=os.path.join(os.getcwd(), ".huggingface", "embedding"),
            )
        case "ollama":
            return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        case "infinity":
            return InfinityEmbeddingsLocal(
                model=EMBEDDING_MODEL_NAME,
                # best to keep at 32
                batch_size=32,
                # for AMD/Nvidia GPUs via torch
                device=device or EMBEDDING_DEVICE,
            )
        case "infinity-server":
            return InfinityEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                infinity_api_url="http://localhost:7797/v1",
            )
        case _:
            raise NotImplementedError("Embedding provider not supported")
