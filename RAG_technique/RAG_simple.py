
"""
Implements basic Retrieve for processing and quering pdf document

+ key conponents:
- PDF processing and text extraction
- text chunking 
- vector store by faiss and openai embedding
- retrieve  for query document
- evaluation RAG system  
"""

import os
import sys

from dotenv import load_dotenv

from langchain.document_loaders  import PyPDFLoader
from langchain.text_slitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


from RAG_technique.helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_space,
                              get_langchain_embedding_provider,
                              show_context, encode_pdf)

from RAG_technique.evaluation.evalute_rag import evaluate_rag
load_dotenv()


def process(path, test_query ):

    chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
    
    context = retrieve_context_per_question(test_query, chunks_query_retriever)
    show_context(context)

    return chunks_query_retriever


if __name__ == "__main__":
    path = ""
    test_query = "What is the main cause of climate change?"
    test = process(path=path, test_query=test_query)

    evaluate_rag(test)