import os
from typing import List

from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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
    "https://sebastianraschka.com/blog/2025/first-look-at-reasoning-from-scratch.html",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
docs_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=docs_splits, collection_name="rag-chroma", embedding=embedding_model
)

retriever = vectorstore.as_retriever()

####  DATA MODEL####


class GradeDocument(BaseModel):
    """Binary score for relevance of retrieved documents"""

    binary_score: str = Field(
        description="Documents are relevant to the question 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocument)

system = """ you are a grader assesing the relevance of documents to a user question.
        If the documents contains keywords or semantic meaning related to the question
        grade it as relevant. Use a binary score: 'yes' for documents that are relevants,
        otherwise 'no' """

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Retrieved_documents: \n\n {document}, \n\n\ User question: {question}",
        ),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[3].page_content
response = retrieval_grader.invoke({"question": question, "document": doc_txt})
print(response)

prompt = hub.pull("rlm/rag-prompt")
print(prompt)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

generation = rag_chain.invoke(
    {
        "context": docs,
        "question": question,
    }
)

##DATA MODEL###


class GradeHallucinations(BaseModel):
    """binary score for the hallucination in the generated anser"""

    binary_score: str = Field(
        description="Answer if the answer is grounded: 'yes' or 'no' "
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system_all = """you are a grader assesing if an LLM answer is supported/ is grounded in a set of retieved facts.\n
        give a binary score of 'yes' or 'no'. 'yes' if the anwer i grounded, 'no' otherwise 
    """
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_all),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation} "),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke(
    {
        "documents": docs,
        "generation": generation,
    }
)


class GradeAnswer(BaseModel):
    """binary score ro assess answer addresses question"""

    binary_score: str = Field(description="Answer addresses question: 'yes' or 'no' ")


structured_llm_grader = llm.with_structured_output(GradeAnswer)

system_answer = """you are a grader assesing if the answer resolves the questiion.\n
        give a binary score of 'yes' or 'no'. 'yes' if the answer resolves the questiom, 'no' otherwise 
    """
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_answer),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation} "),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke(
    {
        "documents": docs,
        "generation": generation,
    }
)

#### question rewriter

system_re = """you are an expert to rewrite question to a better version that is optimized for vector retival.\n
Look at the original question and try to reaon to underlying smantic intent/meaning
"""
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_re),
        (
            "human",
            "here the original question:\n\n {question} \n formulate the optimized version",
        ),
    ]
)
question_rewriter = rewrite_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})


class State(TypedDict):
    question: str
    generation: str
    documents: List[str]


##### NODES


def retrieve(state):
    """
    Retrieve documents
    Args:
        state(dict): The current graph state
    Return:
         state(dict): new key added to the state, documents that contain retrieved documents
    """
    print("--RETRIEVE--")
    question = state["question"]
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
    }


def generate(state):
    """
    Generate answer
    Args:
        state(dict): the current state of the graph
    Returns:
        state(dict): new key added to the state,generation, that contains the LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke(
        {
            "context": documents,
            "question": question,
        }
    )
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines wheter retrieved documents are relevant to the question
    Args:
        state(dict): The current graph state
    Returns:
        state(dict): new key added to the state, only filfered/relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO THE QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {
                "question": question,
                "document": d.page_content,
            }
        )
        grade = score.binary_score
        if grade == "yes":
            print("--Documents are relevants--")
            filtered_docs.append(d)
        else:
            print("--documents are not relevant")
            continue
    return {
        "documents": filtered_docs,
        "question": question,
    }


def transform_query(state):
    """
    Transform query to produce a better question
    Args:
        state(dict): the current state of the graph
    Return:
        state(dict): update question key with optimized question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {
        "documents": documents,
        "question": better_question,
    }


####EDGES
def decide_to_generate(state):
    """
    Determine to generate an answer or reformulate the question
    Args:
        state(dict): the current state of the graph
    Return:
        str: binary decision for next node to call
    """
    print("--ASSES GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]
    if not filtered_documents:
        print("Documents are not relevant")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines if the generation is grounded in the LLM generation
    Args:
        state(dict): the current state of the graph
    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINACTIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    grade = score.binary_score

    if grade == "yes":
        print("---generation is grounded---")

        print("--grade generation vs question")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("decision: generation addresses question")
            return "useful"
        else:
            print("decision: generation doesn t address the question")
            return "not useful"
    else:
        print("genneration is not grounded")
        return "not supported"
