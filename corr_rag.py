import os
from typing import List

from dotenv import load_dotenv
from langchain import hub
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

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

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
vectostore = Chroma.from_documents(
    documents=doc_splits, collection_name="rag-chroma", embedding=embedding_model
)

retriever = vectostore.as_retriever()

"""DATA MODEL"""


class GradeDocuments(BaseModel):
    """binary score for relevane on retrieved docs"""

    binary_score: str = Field(
        description="documents are relevant for the question: 'yes' or 'no' "
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """ you are a grader assesing the relevance of documents to a user question.
        If the documents contains keywords or semantic meaning related to the question
        grade it adìs relevant. Use a binary score: 'yes' for documents that are relevants,
        otherwise 'no'
        """
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "retrieved documents: \n\n {document} \n\n user question: {question}",
        ),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)

doc_txt = docs[2].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

""" GENERATE"""

prompt = hub.pull("rlm/rag-prompt")

rag_chain = prompt | llm | StrOutputParser()
generation = rag_chain.invoke(
    {"context": docs, "question": question}
)  # docs = retriever.invoke(question)
print(generation)

"""QUESTION REWRITER"""

system = """you are a question rewriter that converts an input question to a better question
optimized for web search. Look at inpute question and reason about the semantic meaning. 
"""
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "here is the initial input question: {question}\n Formulate an improved version",
        ),
    ]
)
question_rewriter = rewrite_prompt | llm | StrOutputParser()
new_question = question_rewriter.invoke({"question": question})
print(new_question)

web_search_tool = TavilySearchResults(k=3)

"""GRAPH"""


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]


def retrieve(state):  ##using retriever=vectorstor.as_retriever()
    """
    Retrieve documents

    Args:
        state(dict): The current state of the graph
    Return:
        state(dict): new key added to the state: documents retrieved
    """
    print("---RETRIEVE---")

    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer
    Args:
        state(dict): The current graph state
    Return:
        state(dict): New key added to the state, generation, that contains the LLM answer

    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determine whether retrived documents are relevants
    Args:
        state(dict): the current state of the graph
    Return:
        state(dict): update documents key with only filtered relevant document
    """
    print("---CHECK RELEVANT DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "no"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "yes"
            continue


def transform_query(state):
    """
    Transform query to produce a better question
    Args:
        state(dict): the current graphe state
    Return:
        state(dict): update question key with an optimized question

    """
    print("---TRANSFORM QUESTION---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    web search based on optimized question
    Args:
        state(dict): the current state of the graph
    Returns:
        state(dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")

    # web_search_tool = TavilySearchResults(k=3)
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

'''EDGES'''

def decide_to_generate(state):
    '''
    determine whether to generate an answer or regenerate a question
    Args: 
        state(dict): the current state of the graph
    Returns:
        str: Binary decision for next node to call
    '''

    print ('---ASSESS GRADED DOCUMENTS---')
    state['question']
    web_search=state['web_search']
    state['documents']
    
    if web_search=='yes':
        print(
            '---no documents relevant for the quetion: reformulate quetion for web search'
        )
        return 'transform_query'
    else:
        print('---Generate---')
        return 'generate'

workflow=