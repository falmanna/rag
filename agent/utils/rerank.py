import os

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from agent.utils.lib.infinity_reranker import InfinityCrossEncoder
from configs import EMBEDDING_DEVICE, RERANKING_MODEL_NAME, RERANKING_PROVIDER


def get_reranker(embedding_device: str = None, limit: int = 5):
    match RERANKING_PROVIDER:
        case "infinity":
            model = InfinityCrossEncoder(
                model=RERANKING_MODEL_NAME,
                infinity_api_url="http://localhost:7997",
            )
        case "huggingface":
            model = HuggingFaceCrossEncoder(
                model_name=RERANKING_MODEL_NAME,
                model_kwargs={
                    "device": embedding_device or EMBEDDING_DEVICE,
                    "automodel_args": {
                        "cache_dir": os.path.join(
                            os.getcwd(), ".huggingface", "embedding"
                        ),
                    },
                },
            )
        case _:
            raise NotImplementedError("Reranking provider not supported")

    return CrossEncoderReranker(
        model=model,
        top_n=limit,
    )
