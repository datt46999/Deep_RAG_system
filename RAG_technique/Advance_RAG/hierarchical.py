"""
Implements a Hierarchical index system for document retrieval, utilizing two level for encoding: document level summaries and detail chunking.
This approach aim to improve the efficiency and relevance of information retrieval by the first identifying through summaries, 
then drilling down to specific details within those sections.


Tranditional flat indexing methods can struggle with large documents or corpus, protentially missing context or return irrelecavant information.
Hierarchical indexing addressing this by creating a two-tier search system, allowing for more effecient and context aware retrievel


Key components

1. PDF processing and text chunking
2. Asynchronous document summarization using OpenAI's GPT-4
3. Vector store creation for both summaries and detailed chunks using FAISS and OpenAI embeddings
4. Custom hierarchical retrieval function


Metod Detail:

1. The PDF is loaded and split into documents (likely by page).
2. Each document is summarized asynchronously using GPT-4.
3. The original documents are also split into smaller, detailed chunk
4. Two separate vector store and created:
    + One for document-level summararies
    + One for detaled chunk



"""


import os
import asyncio
from dotenv import load_dotenv 



from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_classic.chains.summarize import load_summarize_chain




from helper_functions import *

load_dotenv()

# def encode_pdf(path, chunk_size: int = 1000, chunk_overlap = 200):
#     loader = PyPDFLoader(path)
#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size = chunk_size,
#         chunk_overlap = chunk_overlap,
#         length_function =len
#     )
#     text_splitter = splitter.split_documents(docs)
#     clean_text = replace_t_space(text_splitter)

#     embedding = OpenAIEmbeddings()
#     vectorstores = FAISS.from_documents(clean_text, embedding)
#     return vectorstores


async def encode_pdf_hierarchical(path, chunk_size: int = 1000, chunk_overlap: int = 200,is_string= False ):
    """
    Asynchronously encode PDF book into hierachical vectore store using openai model
    Includes rate limit handling with exponential backoff

    Args:
    path: Path of file PDF
    chunk_size: the desired size of each chunks
    chunk_overlap : the amount of overlap between consecurtive chunks

    return:
    a Typle two FAISS vector store:
    + Docuement summary level
    + detailed chunks
    """
    if not is_string:
        loader = PyPDFLoader(path)
        documents = await asyncio.to_thread(loader.load)
    else:
        splits = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function= len,
            is_separator_regex=False,
        )
        documents = splits.create_documents([path])

    summary_llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0, max_tokens = 4000)
    summary_chain = load_summarize_chain(summary_llm,  chain_type="map_reduce")


    async def summarize_doc(doc):
        """
        Summary single document with rate limit handling
        Args:
            doc: The document to be summarized.
            
        Returns:
            A summarized Document object.
        """
        summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
        summary = summary_output['output_text']
        return Document(
            page_content=summary,
            metadata={"source": path, "page": doc.metadata["page"], "summary": True}
        )

    # process documents in small batch to advoid rate limit
    batch_size = 5
    summaries =[]
    for i in range(0, len(documents), batch_size):
        batch = documents[i: i+batch_size]
        batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
        summaries.extend(batch_summaries)
        await asyncio.sleep(1)

    # split documents into detail chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len
    )
    
    detail_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)
    # Updated metadata for detailed chunks
    for i, chunk in enumerate(detail_chunks):
        chunk.metadata.update(
            {"chunk_id":i,
            "summary": False,
            "page":int(chunk.metadata.get("page", 0))
            }
        )

    embedding = OpenAIEmbeddings()


    async def create_vectorstore(docs):
        """
        Creates a vector store from a list of documents with rate limit handling.
        
        Args:
            docs: The list of documents to be embedded.
            
        Returns:
            A FAISS vector store containing the embedded documents.
        """
        return await retry_with_exponential_backoff(
            asyncio.to_thread(FAISS.from_documents, docs, embedding)
        )
    summary_vectorstore, detailed_vectorstore = await asyncio.gather(
        create_vectorstore(summaries),
        create_vectorstore(detail_chunks)
    )

    return summary_vectorstore, detailed_vectorstore

def retrieve_hierarchical(query, summary_vectorstore, detail_vectorstore, k_simmarize = 3, k_chunks = 1):
    """
    Performs a hierarchical retrieval using the query.

    Args:
        query: The search query.
        summary_vectorstore: The vector store containing document summaries.
        detailed_vectorstore: The vector store containing detailed chunks.
        k_summaries: The number of top summaries to retrieve.
        k_chunks: The number of detailed chunks to retrieve per summary.

    Returns:
        A list of relevant detailed chunks.
    """

    top_summaries = summary_vectorstore.similarity_search(query,k =k_simmarize)
    relevant_chunks = []
    for summary in top_summaries:
        # for each summary, retrieval detail chunks
        page_number = summary.metadata["page"]
        page_filter = lambda metadata: metadata['page'] == page_number
        page_chunks = detail_vectorstore.similarity_search(query, filter =page_filter, k = k_chunks)
        relevant_chunks.extend(page_chunks)
    return relevant_chunks

async def process():
    path = "data/Understanding_Climate_Change.pdf"


    if os.path.exists("../vector_stores/summary_store") and os.path.exists("../vector_stores/detailed_store"):
        embeddings = OpenAIEmbeddings()
        summary_store = FAISS.load_local("../vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
        detailed_store = FAISS.load_local("../vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)

    else:
        summary_store, detailed_store = await encode_pdf_hierarchical(path)
        summary_store.save_local("vector_stores/summary_store")
        detailed_store.save_local("vector_stores/detailed_store")


    query = "What is the greenhouse effect?"
    results = retrieve_hierarchical(query, summary_store, detailed_store)

    # Print results
    for chunk in results:
        print(f"Page: {chunk.metadata['page']}")
        print(f"Content: {chunk.page_content}...")  # Print first 100 characters
        print("---")


asyncio.run(process())