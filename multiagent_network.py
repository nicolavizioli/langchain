import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("OPENAI_LLM_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_type=os.getenv("OPENAI_API_TYPE"),
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm = chat

tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL


@tool
def python_repl_tool(
    code: Annotated[str, "the python code to execute to generate your code"],
):
    """
    Use this tool to execute Python code. If you want to see the output of a value use the command 'print()'.
    This will be visible to the user.
    """
    # Sostituisci plt.show() con plt.savefig(...)
    if "plt.show()" in code:
        code = code.replace(
            "plt.show()", 'plt.savefig("output.png")  # Automatically saved'
        )

    try:
        result = repl.run(code)
    except BaseException as e:
        return f"failed to execute. Error: {repr(e)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nOut: {result}"
    return (
        result_str
        + "\n\n[INFO] If a chart was generated, it has been saved as `output.png`. You can open it in VS Code."
        + "\nIf you have completed all the task respond: FINAL ANSWER"
    )


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


def get_next_node(
    last_message: BaseMessage, goto: str
):  ##base message puo essere human message, ai message , system
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto


research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=make_system_prompt(
        "you need only to do web research. you wort with a chart generator collegue"
    ),
)


def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], goto="chart_generator")
    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return Command(
        update={
            "messages": result["messages"],
        },
        goto=goto,
    )


chart_agent = create_react_agent(
    llm,
    tools=[python_repl_tool],
    prompt=make_system_prompt(
        "you can only generate prompt, you are working with a reasearcher collegue"
    ),
)


def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], goto="researcher")
    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            "messages": result["messages"],
        },
        goto=goto,
    )


workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

workflow.add_edge(START, "researcher")

graph = workflow.compile()

events = graph.stream(
    {
        "messages": [
            (
                "user",
                "First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                "Once you make the chart, finish.",
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)
for s in events:
    print(s)
    print("----")
