import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI

load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv(
        "AZURE_OPENAI_ENDPOINT"
    ),  # Use azure_endpoint instead of openai_api_base
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("OPENAI_LLM_DEPLOYMENT_NAME"),  # deployment_name is fine
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_type=os.getenv("OPENAI_API_TYPE"),
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY")
)

model = chat
embeddings = embedding_model

from typing import Annotated

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """Messages have the type "list". The `add_messages` function
    in the annotation defines how this state key should be updated
    (in this case, it appends messages to the list, rather than overwriting them)
    """

    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

### add chatbot node


def chatbot(state: State) -> dict:
    response = chat.invoke(state["messages"])
    return {"messages": [response]}


graph_builder.add_node("chatbot", chatbot)

"""chatbot node function takes the current State as input and returns a dictionary containing an 
updated messages list under the key "messages". This is the basic pattern for all LangGraph node functions."""


graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

""" evento è un diz:
        {
    "chatbot": {
        "messages": [lista di messaggi aggiornata]
    }
}
"""
