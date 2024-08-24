import os
from typing import Literal

import instructor

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
            from openai import OpenAI

            return instructor.from_openai(
                OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                ),
                mode=instructor.Mode.JSON,
            )
        case "openai":
            from openai import OpenAI

            return instructor.from_openai(OpenAI(), mode=instructor.Mode.TOOLS)
        case "cohere":
            from cohere import Client

            return instructor.from_cohere(
                Client(), mode=instructor.Mode.COHERE_JSON_SCHEMA
            )
        case "together":
            from openai import OpenAI

            return instructor.from_openai(
                OpenAI(
                    base_url="https://api.together.xyz/v1",
                    api_key=os.environ["TOGETHER_API_KEY"],
                ),
                mode=instructor.Mode.TOOLS,
            )
        case "groq":
            from groq import Groq

            return instructor.from_groq(
                Groq(api_key=os.environ.get("GROQ_API_KEY")), mode=instructor.Mode.TOOLS
            )
        case "fireworks":
            from openai import OpenAI

            return instructor.from_openai(
                OpenAI(
                    base_url="https://api.fireworks.ai/inference/v1",
                    api_key=os.environ["FIREWORKS_API_KEY"],
                ),
                mode=instructor.Mode.TOOLS,
            )

        case _:
            raise ValueError(f"Unknown LLM provider: {provider}")
