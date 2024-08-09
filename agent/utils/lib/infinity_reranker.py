from typing import Any, Dict, List, Tuple

import requests
from langchain_community.cross_encoders.base import BaseCrossEncoder
from langchain_community.embeddings.infinity import (
    TinyAsyncOpenAIInfinityEmbeddingClient,
)
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env


class InfinityRerankClient(TinyAsyncOpenAIInfinityEmbeddingClient):
    def _kwargs_post_request_rerank(
        self, model: str, query: str, documents: List[str]
    ) -> Dict[str, Any]:
        return dict(
            url=f"{self.host}/rerank",
            headers={
                "content-type": "application/json",
            },
            json=dict(
                query=query,
                documents=documents,
                model=model,
            ),
        )

    def _sync_request_rerank(
        self, model: str, query: str, documents: List[str]
    ) -> List[Dict[str, Any]]:
        response = requests.post(
            **self._kwargs_post_request_rerank(
                model=model, query=query, documents=documents
            )
        )
        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}")
        return response.json()["results"]

    def rerank(
        self, model: str, query: str, documents: List[str]
    ) -> List[Dict[str, Any]]:
        return self._sync_request_rerank(model=model, query=query, documents=documents)


class InfinityCrossEncoder(BaseModel, BaseCrossEncoder):
    model: str
    "Underlying Infinity model id."

    infinity_api_url: str = "http://localhost:7997"
    """Endpoint URL to use."""

    client: Any = None  #: :meta private:
    """Infinity client."""

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        values["infinity_api_url"] = get_from_dict_or_env(
            values, "infinity_api_url", "INFINITY_API_URL"
        )

        values["client"] = InfinityRerankClient(
            host=values["infinity_api_url"],
        )
        return values

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        scores = self.client.rerank(
            self.model, text_pairs[0][0], [d[1] for d in text_pairs]
        )

        return [d["relevance_score"] for d in scores]
