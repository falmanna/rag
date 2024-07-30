from typing import Any

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.misc import print_with_time


class Reject(BaseNode):
    @classmethod
    def get_name(cls) -> str:
        return "reject"

    @classmethod
    def invoke(cls, state: GraphState) -> dict[str, Any]:
        print_with_time("---REJECT---")

        rejection = """
        السؤال خارج حدود معرفتي, تخصصي هو الاجابة على اسئلة متعلقة بموسوعة ويكيبيديا باللغة العربية فقط. \n
        يرجى تقديم سؤال آخر."""

        return {"generation": rejection}
