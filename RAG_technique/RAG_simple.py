
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

from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


from helper_functions import (EmbeddingProvider,
                              retrieve_context_per_question,
                              replace_t_space,
                              get_langchain_embedding_provider,
                              show_context, encode_pdf)

from evaluation.evalute_rag import evaluate_rag
load_dotenv()

def encode_pdf(path, chunk_size = 1000, chunk_overlap = 200):
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )

    texts = splitter.split_documents(docs)
    clean_texts = replace_t_space(texts)


    embeddings =  get_langchain_embedding_provider(EmbeddingProvider.OPENAI)

    vectorstore = FAISS.from_documents(clean_texts, embeddings)
    return vectorstore


if __name__ == "__main__":
    path = "data/Understanding_Climate_Change.pdf"
    test_query = "What is the main cause of climate change?"
    chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap= 200)
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
    test_query = "What is the main cause of climate change?"
    context = retrieve_context_per_question(test_query, chunks_query_retriever)
    show_context(context)

    eval_result = evaluate_rag(chunks_query_retriever)
    # from langchain_openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings()
    # print(embeddings.model  )
    import json
    output_path = f"Eval_simple_rag.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    print(f"Eval results saved to {output_path}")





# from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
# loader = DirectoryLoader(
#     path = "../paper_vis",
#     glob = "**/*.pdf",
#     loader_cls = UnstructuredFileLoader,
#     show_progress = True,
#     use_multithreading= True
# )

    