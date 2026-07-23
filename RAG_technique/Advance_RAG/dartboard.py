"""
The DartBoard RAG process addresses a common challenge in large knowledge base: ensuring the retrieved information in both relevant and no-relevant.
By explicitly optimizing a combined relevance-diversity scoring function, it prevents top-k documents from offering same information.

Key components:
1. Relevance and diversoty combination:
    Computes a score factoring in both how pertinent a document is to the query and how distinci it is from already chosen documents.
2. Weighted belancing:
    Introduces RELEVENCE_WEIGHT and DIVERSITY_WEIGHT to allow the dymatic control of scoring
    Help in  avoidiong overly diverse but less relevance results
3. Production ready code:
    Derived from the official implementation yet reorganized for clarity.
    Allows easier integration into existing RAG pipelines.



Method Details:
1. Document Retrieval
 +Obtain an initial set of candidate documents based on similarity (e.g., cosine or BM25).
 +Typically retrieves top-N candidates as a starting point.

2. Scoring and selection
    + Each documents overall score combines relevance and diversity:
    + Select the highest-scoring document, then penalize documents that are overly similar to it
    + Repeat until top-k documents are identified.
3. Hybrid / Fusion & Cross-Encoder Support
    + Essentially: distance between documet and query, and distance between documents
    - For Hybrid/Fusion: Merge similarities(dense and sparse/BM25) in to singe distance.
        Achievement by combine consin ombining cosine similarity over the dense and the sparse vectors (e.g. averaging them).   
        the move to distances is straightforward (1 - mean cosine similarity)
    - Cross-encoders directly use the cross-encoder similarity scores (1- similarity), potentially adjusting with scaling factors.

4. Balancing & Adjustment
    Tune DIVERSITY_WEIGHT and RELEVANCE_WEIGHT based on your needs and the density of your dataset

"""


import os
import numpy as np
from scipy.special import logsumexp
from typing import Tuple, List, Any
from dotenv import load_dotenv

from langchain_core.document_loaders import PyPDFLoader
from langchain_community.vectorstore import FAISS
from langchain_openai import OpenAIEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helper_functions import *

load_dotenv()

DIVERSITY_WEIGHT = 1.0  # Weight for diversity in document selection
RELEVANCE_WEIGHT = 1.0  # Weight for relevance to query
SIGMA = 0.1  # Smoothing parameter for probability distribution
path = "data/Understanding_Climate_Change.pdf"


def conder_pdf(path, chunk_size = 1000, chunk_overlap=200)-> FAISS:
    loader = PyPDFLoader(path)
    docs = loader.load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    text_splitter = splits.split_documents(docs)
    clean_text = replace_t_space(text_splitter)

    # Create embeddings (Tested with OpenAI and Amazon Bedrock)
    embeddings = get_langchain_embedding_provider(EmbeddingProvider.OPENAI)
    #embeddings = get_langchain_embedding_provider(EmbeddingProvider.AMAZON_BEDROCK)
    vectorestore = FAISS.from_documents(clean_text, embeddings)
    return vectorestore
chunks_vectorstore = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
def idx_to_text( idx:int ):
    """Convert a Vector store into corresponding text"""
    docstore_id = chunks_vectorstore.index_to_docstore_id[idx]
    document = chunks_vectorstore.docstore.search(docstore_id)
    return document.page_content
def get_context( query:str,k:int=5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Retrieve top k context items for a query using top k retrieval.
    """
    # regular top k retrieval
    q_vec=chunks_vectorstore.embedding_function.embed_documents([query])
    _,indices=chunks_vectorstore.index.search(np.array(q_vec),k=k)

    texts = [idx_to_text(i) for i in indices[0]]
    return texts


def lognorm(dist:np.ndarray, sigma:float):
    """
    Calculate the log-normal probability for a given distance and sigma.
    """
    if sigma < 1e-9: 
        return -np.inf * dist
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)


def greedy_doartsearch(
    query_distances: np.ndarray,
    document_distances: np.ndarray,
    documents: List[str],
    num_results: int, 
)-> Tuple[List[str], List[str]]:
    """
    Perform greedy dartboard search to select top k document balancing relevance and diversity

   Args:
        query_distances: Distance between query and each document
        document_distances: Pairwise distances between documents
        documents: List of document texts
        num_results: Number of documents to return
    
    Returns:
        Tuple containing:
        - List of selected document texts
        - List of selection scores for each document
    """

    # Avoid division by zero in probability calculations
    sigma = max(SIGMA, 1e-5)
    
    # Convert distances to probability distributions
    query_probabilities = lognorm(query_distances, sigma)
    document_probabilities = lognorm(document_distances, sigma)
    
    # Initialize with most relevant document
    
    most_relevant_idx = np.argmax(query_probabilities)
    selected_indices = np.array([most_relevant_idx])
    selection_scores = [1.0] # dummy score for the first document
    # Get initial distances from the first selected document
    max_distances = document_probabilities[most_relevant_idx]
    
    # Select remaining documents
    while len(selected_indices) < num_results:
        # Update maximum distances considering new document
        updated_distances = np.maximum(max_distances, document_probabilities)
        
        # Calculate combined diversity and relevance scores
        combined_scores = (
            updated_distances * DIVERSITY_WEIGHT +
            query_probabilities * RELEVANCE_WEIGHT
        )
        
        # Normalize scores and mask already selected documents
        normalized_scores = logsumexp(combined_scores, axis=1)
        normalized_scores[selected_indices] = -np.inf
        
        # Select best remaining document
        best_idx = np.argmax(normalized_scores)
        best_score = np.max(normalized_scores)
        
        # Update tracking variables
        max_distances = updated_distances[best_idx]
        selected_indices = np.append(selected_indices, best_idx)
        selection_scores.append(best_score)
    
    # Return selected documents and their scores
    selected_documents = [documents[i] for i in selected_indices]
    return selected_documents, selection_scores

def get_context_with_dartboard(
    query: str,
    num_results: int = 5,
    oversampling_factor: int = 3,
    
) -> Tuple[List[str], List[float]]:
    """
    Retrieve most relevant and diverse context items for a query using the dartboard algorithm.
    
    Args:
        query: The search query string
        num_results: Number of context items to return (default: 5)
        oversampling_factor: Factor to oversample initial results for better diversity (default: 3)
    
    Returns:
        Tuple containing:
        - List of selected context texts
        - List of selection scores
        
    Note:
        The function uses cosine similarity converted to distance. Initial retrieval 
        fetches oversampling_factor * num_results items to ensure sufficient diversity 
        in the final selection.
    """
    # Embed query and retrieve initial candidates
    query_embedding = chunks_vectorstore.embedding_function.embed_documents([query])
    _, candidate_indices = chunks_vectorstore.index.search(
        np.array(query_embedding),
        k=num_results * oversampling_factor
    )
    
    # Get document vectors and texts for candidates
    candidate_vectors = np.array(
        chunks_vectorstore.index.reconstruct_batch(candidate_indices[0])
    )
    candidate_texts = [idx_to_text(idx) for idx in candidate_indices[0]]
    
    # Calculate distance matrices
    # Using 1 - cosine_similarity as distance metric
    document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)
    query_distances = 1 - np.dot(query_embedding, candidate_vectors.T)
    
    # Apply dartboard selection algorithm
    selected_texts, selection_scores = greedy_doartsearch(
        query_distances,
        document_distances,
        candidate_texts,
        num_results
    )
    
    return selected_texts, selection_scores


if __name__ =="__main__":
    
    
    test_query = "What is the main cause of climate change?"
    texts,scores=get_context_with_dartboard(test_query,k=3)
    show_context(texts)
