from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.runnables import RunnableSerializable

from agent.state import GraphState


class BaseNode(ABC):
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def get_chain(cls) -> Optional[RunnableSerializable]:
        pass

    @classmethod
    @abstractmethod
    def invoke(cls, state: GraphState):
        raise NotImplementedError
