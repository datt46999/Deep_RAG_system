"""
Fusion retrieval aims to combine these methods to create a more robust and accurate retrieval system that can handle a wider range of queries effectively.

Key Components:
1. PDF processing and text chunking
2. vectorstore create using FAISS and OpenAI EMbedding
3. BM25 index creation for keywork-based retrieval
4. Custom fusion retrieval function that combines both methods


"""


import os
import fitz
import numpy as np
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from helper_functions import *


load_dotenv()

# def read_pdf_to_text(path):
#     docs = fitz.open(path)
#     content = ""
#     for page_num in range(len(docs)):
#         page = docs[page_num]
#         content += page.get_text()
#     return content


def encode_pdf_and_get_split_document(path, chunk_size = 1000, chunk_overlap = 200):
    """
    Encode pdf docuemt into vectorstore using OpenAI
    ARG:
    path(str): path to pdf file
    chunk_size (int): the desize of each text chunk
    chunk_overlap (int): the amount of overlap between consecutive chunks

    return:
    A FAISS vector store containing the encoded book content.
    """

    loader = PyPDFLoader(path)
    docs = loader.load()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len
    )
    texts_splitter = splitter.split_documents(docs)
    clean_text = replace_t_space(texts_splitter)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(clean_text, embedding)
    return vectorstore, clean_text


def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create BM25 index from given documents.
    BM25 (Best Marching 25) is ranking function used in information retrieval
    It's base on probabilitic retrieval framework and is an improvement over TF-IDF
    Args:
    documents (List[Document]): List of documents to index.

    Returns:
    BM25Okapi: An index that can be used for BM25 scoring.
    """
    # Tokenize each document by splitting on whitespace
    # This is a simple approach and could be improved with more sophisticated tokenization
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)
    

def fusion_retrieval(vectorstore, bm25, query: str, k : int= 5,alpha :float = 0.5) -> List[Document]:
    """
    Perfrom fusion retrieval combining keywork-based (BM25) and vectorstore search

    Args:
    vectorstore (VectorStore): The vectorstore containing the documents.
    bm25 (BM25Okapi): Pre-computed BM25 index.
    query (str): The query string.
    k (int): The number of documents to retrieve.
    alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
    List[Document]: The top k documents based on the combined scores.
    """

    epsilon = 1e-8
    all_docs = vectorstore.sumilarity_search("", k = vectorstore.index.ntotal)

    bm25_scores = bm25.get_scores(query.split())

    vector_results = vectorstore.similarity_search_with_score(query, k = len(all_docs))

    # Normalize scores
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) -  np.min(bm25_scores) + epsilon)
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores  
    

    # Rank documents
    sorted_indices = np.argsort(combined_scores)[::-1]
    
    
    return [all_docs[i] for i in sorted_indices[:k]]

if __name__ == "__main__":
    path = "data/Understanding_Climate_Change.pdf"

    vectorstore, cleaned_texts = encode_pdf_and_get_split_document(path)
    bm25 = create_bm25_index(cleaned_texts) 
    query = "What are the impacts of climate change on the environment?"

    # Perform fusion retrieval
    top_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)
    docs_content = [doc.page_content for doc in top_docs]
    show_context(docs_content)
