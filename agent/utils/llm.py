from typing import Literal

from langchain_cohere import ChatCohere
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from configs import LLM_MODEL_NAME, LLM_PROVIDER


def get_llm(
    *,
    model: str = LLM_MODEL_NAME,
    provider=LLM_PROVIDER,
    temperature: str = 0,
    format: Literal["", "json"] = "json",
):
    match provider:
        case "ollama":
            return ChatOllama(model=model, temperature=temperature, format=format)
        case "openai":
            return ChatOpenAI(model=model, temperature=temperature)
        case "cohere":
            return ChatCohere(model=model, temperature=temperature)

        case _:
            raise ValueError(f"Unknown LLM provider: {provider}")
