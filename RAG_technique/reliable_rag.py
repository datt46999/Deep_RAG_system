"""
Visual Representaion

Star(query) --> Vectorstore (Retriever doc + query) -->Check_document relevant(relevant docs + query)---\n

----> Generate answer(relevant docs + answer) ---->check Hullucination   (Query + relevant docs + answer) ->  highlight Document Snippet -> End
"""

import os

from dotenv import load_dotenv
from typing import List


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_groq import ChatGroq


from pydantic import BaseModel, Field



load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') # For LLM -- llama-3.1-8b (small) & mixtral-8x7b-32768 (large)
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')
urls = [
    "https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-4-planning/?ref=dl-staging-website.ghost.io",
    "https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/?ref=dl-staging-website.ghost.io"
]
embedding_model = CohereEmbeddings(model="embed-english-v3.0")
model = "llama-3.1-8b-instant"


class GradeDocuments(BaseModel):
    """Binary search for relevance check on retrieved documents"""
    binary_score :str = Field(
        description = "Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucination(BaseModel):
    """Binary score for hallucination present in 'generaion' answer"""
    binary_score: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""

    id: List[str] = Field(
        ...,
        description="List of id of docs used to answers the question"
    )

    title: List[str] = Field(
        ...,
        description="List of titles used to answers the question"
    )

    source: List[str] = Field(
        ...,
        description="List of sources used to answers the question"
    )

    segment: List[str] = Field(
        ...,
        description="List of direct segements from used documents that answers the question"
    )


def emb2vectorstore(doc_list, chunk_size = 500, chunk_overlap = 0):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    doc_splits = text_splitter.split_documents(doc_list)
    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag",
        embedding=embedding_model,
    )

    retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': 4}, # number of documents to retrieve
                )
    return retriever

def format_docs(docs):
    return "\n".join(f"<doc{i+1}>:\nTitle:{doc.metadata['title']}\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))



def relevency_docs():
    llm = ChatGroq(model =model, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """
        You are grader assessing relevance of a retrieved document to a user question.\n
        If the document contains keysword(s) or semantic meaning related to the user question, grade is a relevant.\n
        It does not need to be  a stringent test. The goal is to filter out the erroneous retrievals.\n
        Give binary score 'yes' or 'no' score indicate whether the document is relevant  to the question.    
    """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system),
            ('human', 'Retrieved document: \n\n {documents} \n\n User question: {question}')
        ]
    ) 
    
    return grade_prompt | structured_llm_grader

    


def generate_result():
    llm = ChatGroq(model=model, temperature=0)
    system = """
        You are assisstant for question-answering tasks. Answer the questions base upon your knowledge.
        user three-to-five sentences maximum and keep the answer concise
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system),
            ('human', "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
        ]
    )
    return prompt | llm | StrOutputParser()
    

def check_hallucination():
    llm = ChatGroq(model= model, temperature =0)
    structure_llm_grader = llm.with_structured_output(GradeHallucination)
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", system),
        ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
        ]
    )
    return hallucination_prompt | structure_llm_grader


def highlight_docs():
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

    # parser
    parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

    # Prompt
    system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
    1. A question.
    2. A generated answer based on the question.
    3. A set of documents that were referenced in generating the answer.

    Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
    generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
    in the provided documents.

    Ensure that:
    - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
    - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
    - (Important) If you didn't used the specific document don't mention it.

    Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

    <format_instruction>
    {format_instructions}
    </format_instruction>
    """


    prompt = PromptTemplate(
        template= system,
        input_variables=["documents", "question", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Chain
    return prompt | llm | parser



if __name__ == "__main__":
    docs = [WebBaseLoader(url).load() for url in urls ]
    doc_list = [item for sublist in docs for item in sublist]
    # print(embedding_model.model)    

    question = "what are the differnt kind of agentic design patterns?"
    retriever = emb2vectorstore(doc_list)

    docs = retriever.invoke(question)
    print(f"Title: {docs[0].metadata['title']}\n\n Source: {docs[0].metadata['source']} \n\n Content:  {docs[0].page_content}")
    print("=="*50)
    print("\n\n")

    retrieval_grader  = relevency_docs()
    docs_to_use = []
    for doc in docs:
        print(doc.page_content, '\n', '-'*50)
        res = retrieval_grader.invoke({"question": question, "documents": doc.page_content})
        print(res,'\n')
        if res.binary_score == 'yes':
            docs_to_use.append(doc)


    print("=="*50)
    print("\n\n")
    rag_chain = generate_result()
    generation = rag_chain.invoke({"documents":format_docs(docs_to_use), "question": question})
    print(generation)
    hallucination_grader = check_hallucination()
    response = hallucination_grader.invoke({"documents": format_docs(docs_to_use), "generation": generation})
    print("=="*50)
    print("\n\n")
    print(response)
    doc_lookup = highlight_docs()
    lookup_response = doc_lookup.invoke({"documents":format_docs(docs_to_use), "question": question, "generation": generation})


    for id, title, source, segment in zip(lookup_response.id, lookup_response.title, lookup_response.source, lookup_response.segment):
        print(f"ID: {id}\nTitle: {title}\nSource: {source}\nText Segment: {segment}\n")