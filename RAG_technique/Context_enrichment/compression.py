"""
This code demonstrates the implementation of contextual compression in a document retrieval system using LangChain and OpenAI's language models. 
The technique aims to improve the relevance and conciseness of retrieved information by 
compressing and extracting the most pertinent parts of documents in the context of a given query.


Vector store creation from a PDF document
Base retriever setup
LLM-based contextual compressor
Contextual compression retriever
Question-answering chain integrating the compressed retriever

"""


import os

from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI


from helper_functions import *
load_dotenv()


def retriever_compressor(path):
    vector_store = encode_pdf(path)
    retriever = vector_store.as_retriever()

    #Create a contextual compressor
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
    compressor = LLMChainExtractor.from_llm(llm)

    #Combine the retriever with the compressor
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    # Create a QA chain with the compressed retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=compression_retriever,
        return_source_documents=True
    )
    return qa_chain


if __name__ == "__main__":
    path =  "data/Understanding_Climate_Change.pdf"
    qa_chain = retriever_compressor(path)


    query = "What is the main topic of the document?"
    result = qa_chain.invoke({"query": query})
    print(result["result"])
    print("Source documents:", result["source_documents"])
            