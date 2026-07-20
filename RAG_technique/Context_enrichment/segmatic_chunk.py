"""

Semantic chunking represents an advanced approach to document processing for retrieval systems. 
By attempting to maintain semantic coherence within text segments,
 it has the potential to improve the quality of retrieved information and enhance the performance of downstream NLP tasks. 
 This technique is particularly valuable for processing long, 
complex documents where maintaining context is crucial, such as scientific papers, legal documents, or comprehensive reports.
"""

import os
import sys
from dotenv import load_dotenv

# Original path append replaced for Colab compatibility
from helper_functions import *
from evaluation.evalute_rag import *

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

def process(path):
    content = read_pdf_of_string(path)

    """
    percentile': all differences between sentences are calculated, and then any difference greater than the X percentile is split.
    'standard_deviation': any difference greater than X standard deviations is split.
    'interquartile': the interquartile distance is used to split chunks.
    """
    # chose which embeddings and breakpoint type and threshold to use
    text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90)
    docs = text_splitter.create_documents([content])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return chunks_query_retriever

if __name__ =="__main__":
    path = "data/Understanding_Climate_Change.pdf"
    test_query = "What is the main cause of climate change?"
    chunks_query_retriever = process(path)
    context = retrieve_context_per_question(test_query, chunks_query_retriever)
    show_context(context)