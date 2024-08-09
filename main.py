from langchain.globals import set_debug, set_verbose

from agent.graph import get_main_graph
from agent.state import GraphConfig
from agent.utils.misc import print_with_time

set_verbose(False)
set_debug(False)


if __name__ == "__main__":
    print_with_time("Starting")
    app = get_main_graph()
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")

    print_with_time(
        app.invoke(
            input={
                "question": "ما هو تاريخ بدء التقويم الهجري، وما هو الحدث الذي تم اعتماده كنقطة انطلاق لهذا التقويم؟"
            },
            config={
                "configurable": GraphConfig(
                    embedding_rerank=True, question_rewriter=True
                ).dict()
            },
        ).get("generation")
    )

    # get_vectorstore_retriever().invoke("هل زار اينشتاين جامعة ليتكولن؟")
