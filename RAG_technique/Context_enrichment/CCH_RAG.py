"""
Contextual Chunk Headers
Keys components
The idea here is to add header in hight level context to the chunk by prepending a chunk header
This chunk header could be as simple as the document title, or it could use a combination of document title,
concise document summary, ans the hierarchy of section  and sub-section titles

"""


import os
import cohere
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from typing import List 

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import FAISS



from helper_functions import *


load_dotenv()


DOCUMENT_TITLE_PROMPT ="""
    What is title of following ducument?
    Your response MUST be the title of the dicument, and nothing else. Do NOT with anthing else.
    {document_title_guidance}
    {truncation_message}
    DOCUMENT
    {document_text}
""".strip()


TRUNCATION_MESSAGE = """
Also note that  the document text provide below is just the first ~{num_words} words of the document.
That should be plenty for this task. Your response should still pretain to the entire document, not just the text provided below.
""".strip()

MAX_CONTENT_TOKENS = 4000
MODEL_NAME = 'gpt-4o-mini'
TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')


def split_into_chunks(text, chunk_size: int = 800):
    """
    Split a given text into specifice list chunk by  RecusiveCharacterTextSplitter
    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 800.

    Returns:
        list[str]: A list of text chunks.

    Example:
        >>> text = "This is a sample text to be split into chunks."
        >>> chunks = split_into_chunks(text, chunk_size=10)
        >>> print(chunks)
        ['This is a', 'sample', 'text to', 'be split', 'into', 'chunks.']
    
        
    split_documents: input is pdf
    create_documents: input is text string
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = 0,
        length_function = len
    )
    documents  = splitter.create_documents([text])
    return [document.page_content for document in documents]



# generate descriptive document titple to use in chunk header

def make_llm_call(chat_messages: list[dict])-> str:
    """
    Make an API call to the OpenAI model
    Args:
        chat_messages (list[dict]): A list of message dictionaries for the chat completion.

    Returns:
        str: The generated response from the language model.
    """
    client = OpenAI(api_key= os.getenv("OPEN_AIP_KEY"))
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=chat_messages,
        max_tokens=MAX_CONTENT_TOKENS,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
def truncate_content(content: str, max_tokens: str) -> tuple[str, int]:
    """
    truncate the content to a specifier maximun number of tokens
    Args:
        content (str): The input text to be truncated.
        max_tokens (int): The maximum number of tokens to keep.

    Returns:
        tuple[str, int]: A tuple containing the truncated content and the number of tokens.
    """

    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
    return TOKEN_ENCODER.decode(truncated_tokens), min(len(tokens), max_tokens)

def get_document_title(document_text: str, document_title_guidance :str = "")-> str:
    """
    Extract title document using a language model
    Args:
        document_text (str): The text of the document.
        document_title_guidance (str, optional): Additional guidance for title extraction. Defaults to "".

    Returns:
        str: The extracted document title.
    """

    document_text, num_token = truncate_content(document_text, MAX_CONTENT_TOKENS)
    truncate_message = TRUNCATION_MESSAGE.format_map(num_words = 3000) if num_token>= MAX_CONTENT_TOKENS else ""


    # prepare the content for title extraction
    prompt = DOCUMENT_TITLE_PROMPT.format(
        document_title_guidance = document_title_guidance,
        document_text= document_text, 
        truncate_messages = truncate_message
    )
    chat_message = [{'role': 'user', 'content':prompt}]
    return make_llm_call(chat_message)

# add chunk header and measure impact
def rerank_documents(query: str, chunks:list[str]) -> list[float]:
    """
    Use Cohere Rerank API to rerank the search results
    args: 
    query (str): the search query
    chunks (List[str]):List of document chunk to reranked

    return:
    List([float]):List of similarity score of each chunk, in the original order
    """
    MODEL = "rerank-english-v3.0"
    client = cohere.Client(api_key = os.environ["CO_API_KEY"])

    rerank_results = client.rerank(model = MODEL, query = query, documents = chunks)
    results = rerank_results.results
    reranked_ids = [result for result in results]
    reranked_similarity_score = [result.relevant_score for result in results]

    similary_score = [0]*len(chunks)
    for i, index  in enumerate(reranked_ids):
        similary_score[index] = reranked_similarity_score[i]
    return similary_score


def compare_chunk_similarities(chunk_index: int, chunks: List[str], document_title: str, query: str) -> None:
    """
    Compare similarity scores for a chunk with and without a contextual header.

    Args:
        chunk_index (int): Index of the chunk to inspect.
        chunks (List[str]): List of all document chunks.
        document_title (str): Title of the document.
        query (str): The search query to use for comparison.

    Prints:
        Chunk header, chunk text, query, and similarity scores with and without the header.
    """
    chunk_text = chunks[chunk_index]
    chunk_wo_header = chunk_text
    chunk_w_header = f"Document Title: {document_title}\n\n{chunk_text}"

    similarity_scores = rerank_documents(query, [chunk_wo_header, chunk_w_header])

    print(f"\nChunk header:\nDocument Title: {document_title}")
    print(f"\nChunk text:\n{chunk_text}")
    print(f"\nQuery: {query}")
    print(f"\nSimilarity without contextual chunk header: {similarity_scores[0]:.4f}")
    print(f"Similarity with contextual chunk header: {similarity_scores[1]:.4f}")
if __name__ == "__main__":

        # File path for the input document
    FILE_PATH = "data/nike_2023_annual_report.txt"

    # Read the document and split it into chunks
    with open(FILE_PATH, "r") as file:
        document_text = file.read()

    chunks = split_into_chunks(document_text, chunk_size=800)
    document_title = get_document_title(document_text)
    CHUNK_INDEX_TO_INSPECT = 86


    QUERY = "Nike climate change impact"

    compare_chunk_similarities(CHUNK_INDEX_TO_INSPECT, chunks, document_title, QUERY)
