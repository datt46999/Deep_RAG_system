"""
Context enrichment window for document retriever
Implement a context enrichment window technique for document retrieval in vector database.
Enchance the standard retrieval process by adding surrounding  context to each retrieved chunk


Motivation:
Traditional vector search often returns isolated chunks of text, which may lack necessary context for full understanding. 
This approach aims to provide a more comprehensive view of the retrieved information by including neighboring text chunks.


Key components:
1. PDF processing and text chunking
2. Vector store creation using FAISS and OpenAI embeddings
3. Custom retrieval function with context window
4. Comparison between standard and context-enriched retrieval


"""
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from helper_functions import *

load_dotenv()

chunk_size = 400
chunk_overlap =200

def split_text_to_chunk_with_indices(text: str, chunk_size: int, chunk_overlap: int)-> list[Document]:
    """
    Split text into chunks with metadata of chunk chronological index
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content = chunk, metadata= {"index": len(chunks),"text": text}))
        start += chunk_size - chunk_overlap
    return chunks 

def get_chunk_by_index(vectorstore, targer_index: int)-> Document:
    """
    Retrieve a chunk from the vectostore base on its index in the metadata

    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    target_index (int): The index of the chunk to retrieve.
    
    Returns:
    Optional[Document]: The retrieved chunk as a Document object, or None if not found.
    """

    # This is a simplified version. In practice, you might need effect menthob to retrieve chunk index.
    # depend on vectorstore on you implementation
    all_docs = vectorstore.similarity_search("", k = vectorstore.index.ntotal)
    for doc in all_docs:
        if doc.metadata.get("index") == targer_index:
            return doc
    return None


def retriever_with_context_overlap(vectorstore, retriever, query: str, 
                                   num_neighbor: int = 1, chunk_size: int = 200, chunk_overlap: int =20)-> list[str]:
    """
    Retriever chunks base on query, then fetch neighboring chunk and concatenate them,
    accouting for overlap and correct indexing
    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    retriever: The retriever object to get relevant documents.
    query (str): The query to search for relevant chunks.
    num_neighbors (int): The number of chunks to retrieve before and after each relevant chunk.
    chunk_size (int): The size of each chunk when originally split.
    chunk_overlap (int): The overlap between chunks when originally split.

    Returns:
    List[str]: List of concatenated chunk sequences, each centered on a relevant chunk.
    """
    retriever_chunks = retriever.invoke(query)
    result_query = []
    for chunk in retriever_chunks:
        current_index = chunk.metadata.get("index")
        if current_index == None:
            continue

        # detimine range for chunk retriever
        start_index = max(0, current_index - num_neighbor)
        end_index = current_index + num_neighbor +1

        # retriever all chunk in the range:
        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:
                neighbor_chunks.append(neighbor_chunk)

        # sort chunks by their index  to ensure correct order
        neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

        # Concatenated chunks, accounting  for overlap
        # Concatenate chunks, accounting for overlap
        concatenated_text = neighbor_chunks[0].page_content
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        result_query.append(concatenated_text)

    return result_query
if __name__ == "__main__":
    path = "data/Understanding_Climate_Change.pdf"
    text = read_pdf_of_string(path)
    docs = split_text_to_chunk_with_indices(text, chunk_size, chunk_overlap)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    chunks_query_retrieve = vectorstore.as_retriever(search_kwargs= {'k':1})

    # chunk = get_chunk_by_index(vectorstore, 0)

    query = "Explain the role of deforestation and fossil fuels in climate change."
    baseline_chunk = chunks_query_retrieve.invoke(query, k=1)
    # Focused context enrichment approach
    enriched_chunks = retriever_with_context_overlap(
        vectorstore,
        chunks_query_retrieve,
        query,
        num_neighbor=1,
        chunk_size=400,
        chunk_overlap=200
    )

    print("Baseline Chunk:")
    print(baseline_chunk[0].page_content)
    print("\nEnriched Chunks:")
    print(enriched_chunks[0])

