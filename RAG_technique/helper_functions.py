import asyncio
import textwrap
import os
import fitz
import numpy as np
import random 

from typing import List

from pydantic import BaseModel, Field

from rank_bm25 import BM25Okapi
from openai import RateLimitError
from enum import Enum

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate


def replace_t_space(list_of_documents):
    """
    Replace all tab characters '\t'with space in the page content of each document
    args: 
    list_of_documents: a list of document object, each with a 'page_content' attribute.
    Return:
    modified list of documents with tab characters replaced by space 
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t'," ")
    return list_of_documents

def text_wrap(text, width = 120):
    """
    Wrap input text to the specified width
    arg:
    text: content in document
    width: width which tp wrap the text

    return:
    the wrapped text
    """
    return textwrap.fill(text, width = width)


def encode_pdf(path, chunk_size = 1000, chunk_overlap = 250):
    """
    Encode PDF input into a vectorsore using OpenAi Embedding
    Arg:
    path: path to file PDF
    chunk_size : the desired chunk size of each text chunk
    chunk_overlap: the amount of overlep between consencutive chunks
    Return:
      A FAISS vector store containingn the encoded books content
    """
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter =  RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    splitter_docs = splitter.split_documents(docs)
    clean_docs= replace_t_space(splitter_docs)

    embedding = OpenAIEmbeddings()
    vectorstore_emd = FAISS.from_documents(clean_docs, embedding)
    return vectorstore_emd
    

def encode_from_string(content, chunk_size = 1000, chunk_overlap = 250):
    """
    Encode a string into a vector store using OpenAi embeddings
    ARG:
    content: the text content to be encoder
    chunk_size: the size of each chunk  of text
    chunk_overlap: the overlap between chunk

    RETURN:
    FAISS: a vector store containing the encoded content

    Raises:

    ValueError: If the input content is not valid.
    RuntimeError: If there is an error during the encoding process.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string")
    if not isinstance(chunk_size, int) or chunk_size<=0:
        raise ValueError("Chunk size must be possible interger")
    if not isinstance(chunk_overlap, int) or chunk_overlap<=0:
        raise ValueError("Chunk overload must be possible interger")
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            is_separator_regex = False
        )
        chunks = splitter.create_documents([content])

        # assign metadata
        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0
        
        # generate embedding and create vectorstore
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embedding)
    
    except Exception as e:
        raise RuntimeError(f"An error occured during the encoding process: {str(e)}")
    return vectorstore

def retrieve_context_per_question(question, chunk_query_retriever):
    """
    Retriever relevant centext and unique URLs for a given question using the chunks query retriever
    ARGs:
    Question: the question for each retrieve context and URLs
    Returns:
    A tuple containing:
    - a string concatenanted content of relvant document
    - s list of unique URLs from the metadata of the relevant document
    """
    # retriver relevant document for the given question
    docs = chunk_query_retriever.invoke(question)
     
    # concatanate document content
    # context = " ".join(doc.page_content for doc in docs)
    context = [doc.page_content for doc in docs]
    return context 

class QuestionAnswerFromContext(BaseModel):
    """
    Model generate an answer to a query base on a given context
    attribute:
    answer_based_on_content: the generated answer base on the context
    """
    answer_base_on_context: str = Field(description="Generates an answer to a query based on a given context.")


def create_question_answer_from_context_chain(llm):
    # using model llm to create question answer from context 
    question_answer_from_context_chain = llm

    question_answer_prompt_template= """
    For the Question below, provide a concise but suffice answer base on the provide context:
    {context}
    Question
    {question}
    """
    
    # create promptTemplate with the specified template and input variable
    question_answer_from_context_prompt = PromptTemplate(
        template = question_answer_prompt_template,
        input_variables = ["context", "question"],
    )


    # create chain by combining the prompt template and the language model
    question_answer_from_context_cot_chain = question_answer_from_context_prompt | question_answer_from_context_chain.with_structured_output(
        QuestionAnswerFromContext
    )
    return question_answer_from_context_cot_chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    Answer a question by using the given context by invoking a chain of reasoning
    arg:
    question : The question to be answered
    context : the context to be use for answering the question

    return:
    a dictionary containing the answer, context, and question

    """
    input_data = {
        "question": question,
        "context": context
    }

    # Answering the question from the retrieved context
    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_base_on_context
    return {"answer": answer, "context": context, "question":question}


def show_context(context):
    """
    display context of the provided context list
    arg:
    context: a list of context items to be displayed
    prints each context item in the list with a heading indicating its position
    """
    for i, c in enumerate(context):
        print(f"context {i}")
        print(c)
        print('\n')

def read_pdf_of_string(path):
    """
    Read a PDF document from the specified path and return its content as a string.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The concatenated text content of all pages in the PDF document.

    The function uses the 'fitz' library (PyMuPDF) to open the PDF document, iterate over each page,
    extract the text content from each page, and append it to a single string.
    """
    # open teh PDF document located at the specified path
    doc = fitz.open(path)
    content = ""

    # iterate over each page in the document
    for page_num in range(len(doc)):
        # get current page
        page = doc[page_num]
        #  extract the text content from the current page and append it to the content string
        content += page.get_text()
    return content

def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query:str, k: int = 5)-> List[str]:
    """
    Perform BM25 retrieval and return the top k cleaned text chunks.

    Args:
    bm25 (BM25Okapi): Pre-computed BM25 index.
    cleaned_texts (List[str]): List of cleaned text chunks corresponding to the BM25 index.
    query (str): The query string.
    k (int): The number of text chunks to retrieve.

    Returns:
    List[str]: The top k cleaned text chunks based on BM25 scores.
    """
    
    query_token = query.split()

    # get BM25 score for the query
    bm25_score = bm25.get_scores(query_token)

    # get the indices of the top k score
    top_k_indicate = np.argsort(bm25_score)[::-1][:k]

    # get top k retriever
    top_k_text = [cleaned_texts[i] for i in top_k_indicate]
    return top_k_text


async def exponential_backoff(attempt):
    """
    Implement exponential backoff with a jitter
    Args:
        attempt: The current retry attempt number.
        
    Waits for a period of time before retrying the operation.
    The wait time is calculated as (2^attempt) + a random fraction of a second.
    """
    # Calculate the wait time with exponential backoff and jitter
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")

    # Asynchronously sleep for the calculated wait time
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    Retries a coroutine using exponential backoff upon encountering a RateLimitError.
    
    Args:
        coroutine: The coroutine to be executed.
        max_retries: The maximum number of retry attempts.
        
    Returns:
        The result of the coroutine if successful.
        
    Raises:
        The last encountered exception if all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            # Attempt to execute the coroutine
            return await coroutine
        except RateLimitError as e:
            # If the last attempt also fails, raise the exception
            if attempt == max_retries - 1:
                raise e

            # Wait for an exponential backoff period before retrying
            await exponential_backoff(attempt)

    # If max retries are reached without success, raise an exception
    raise Exception("Max retries reached")


# Enum class representing different embedding providers
class EmbeddingProvider(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"

# Enum class representing different model providers
class ModelProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"
def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    """
    Returns an embedding provider based on the specified provider and model ID.

    Args:
        provider (EmbeddingProvider): The embedding provider to use.
        model_id (str): Optional -  The specific embeddings model ID to use .

    Returns:
        A LangChain embedding provider instance.

    Raises:
        ValueError: If the specified provider is not supported.
    """

    if provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif provider == EmbeddingProvider.COHERE:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings()
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id) if model_id else BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")