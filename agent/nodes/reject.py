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

        rejection: str
        if not state.question_accepted:
            rejection = "السؤال خارج حدود معرفتي, تخصصي هو الاجابة على اسئلة متعلقة بموسوعة ويكيبيديا باللغة العربية فقط. يرجى تقديم سؤال آخر."
        else:
            rejection = "عذرا, لم استطع ايجاد مصادر مناسبة للسؤال المطروح. يرجى تقديم سؤال آخر او ايضاح السؤال."

        return {"generation": rejection}
