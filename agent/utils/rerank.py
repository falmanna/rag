import os

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from configs import EMBEDDING_DEVICE, RERANKING_MODEL_NAME


def get_reranker(embedding_device: str = None, limit: int = 5):
    return CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(
            model_name=RERANKING_MODEL_NAME,
            model_kwargs={
                "device": embedding_device or EMBEDDING_DEVICE,
                "automodel_args": {
                    "cache_dir": os.path.join(os.getcwd(), ".huggingface", "embedding"),
                },
            },
        ),
        top_n=limit,
    )
