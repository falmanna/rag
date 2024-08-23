import chainlit as cl
from chainlit.input_widget import Switch
from langgraph.graph.state import CompiledStateGraph

from agent.graph import get_main_graph
from agent.state import GraphConfig


@cl.on_settings_update
async def update_settings(settings):
    configs = GraphConfig(
        embedding_rerank=settings["embedding_rerank"],
        llm_rerank=settings["llm_rerank"],
        question_rewriter=settings["question_rewriter"],
        usefulness_grader=settings["usefulness_grader"],
        hallucination_grader=settings["hallucination_grader"],
        summarize_docs=settings["summarize_docs"],
    )

    cl.user_session.set("configs", configs)
    await cl.Message(content="✅ Settings updated successfully.").send()


@cl.on_chat_start
async def on_chat_start():
    settings = cl.ChatSettings(
        [
            Switch(
                id="embedding_rerank", label="Enable embedding rerank", initial=True
            ),
            Switch(id="llm_rerank", label="Enable LLM rerank", initial=False),
            Switch(
                id="question_rewriter", label="Enable question rewrite", initial=True
            ),
            Switch(
                id="usefulness_grader", label="Enable usefulness grader", initial=False
            ),
            Switch(
                id="hallucination_grader",
                label="Enable hallucination grader",
                initial=False,
            ),
            Switch(
                id="summarize_docs",
                label="Enable document summarization (Enable if context windows is small)",
                initial=False,
            ),
        ]
    )
    await settings.send()

    configs = GraphConfig(
        embedding_rerank=settings.settings()["embedding_rerank"],
        llm_rerank=settings.settings()["llm_rerank"],
        question_rewriter=settings.settings()["question_rewriter"],
        usefulness_grader=settings.settings()["usefulness_grader"],
        hallucination_grader=settings.settings()["hallucination_grader"],
        summarize_docs=settings.settings()["summarize_docs"],
    )

    app = get_main_graph()
    cl.user_session.set("app", app)
    cl.user_session.set("configs", configs)


@cl.on_message
async def on_message(msg: cl.Message):
    app: CompiledStateGraph = cl.user_session.get("app")
    configs: GraphConfig = cl.user_session.get("configs")

    response = None
    try:
        async for event in app.astream(
            {"question": msg.content},
            config={"configurable": configs.dict()} if configs else None,
        ):
            print("EVENT:", event)
            for key in event.keys():
                async with cl.Step(name=key) as step:
                    step.output = event[key]
                    await step.update()

                if "generation" in event[key]:
                    response = cl.Message(content=event[key]["generation"])
    except Exception as e:
        print("ERROR:", e)
        response = cl.Message(content="An error occurred, please try again.")
    finally:
        await response.send()
