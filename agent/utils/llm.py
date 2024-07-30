from typing import Literal

from langchain_ollama import ChatOllama

from configs import LLM_MODEL_NAME, LLM_PROVIDER


def get_llm(
    *,
    model: str = LLM_MODEL_NAME,
    provider=LLM_PROVIDER,
    temperature: str = 0,
    format: Literal["", "json"] = "json",
):
    return ChatOllama(model=model, temperature=temperature, format=format)
