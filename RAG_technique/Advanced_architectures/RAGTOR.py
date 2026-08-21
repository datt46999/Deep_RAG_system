"""
RAGTOR: advanced infomation retrieved and question-answer system combine hierachical document summarization, embedding base-on retrieval, and context answer generation.
Is aims to efficiently handle large documents connection.


Key components
1 Tree building: Creates a hierachical structure  of document summaries
2 Embedding and Clustering: Original docunemts and summaries base on semantic similarity
3 Vectorstore: Efficiently stores  and retrieves docment and summary embedding
4 Contextual Retriever: Select the most relevent information for a given query
5 Answer Generation: Produces Coherent response base on retrieved information


Methos Details

Tree building 
    1: Start with original documents as level 0
    2: For each level:
        Embedding text using llm
        Cluster embedding (Using Gaussian Mixture Models)
        Generate summaries for each cluster
        Use these summaries as the text for the next level
    3 Continue until reaching a single summary or maximun levels

Embedding and Retrieval
    1 Embed all documents and summaries from all level of the tree
    2 Store these embeddings in a vectorestore (FAISS) for efficient similarity search
        embed the query
        Retrieve the mose similar documents summaries from the vectorestore

Contextual Compression
    Take the retrieved documents/summaries
    Use LLM to extract only the most relevant parts for the given query

Answer Generation
    Combine the relevant text into context
    use LLM to generate the the answer base in this context and user query

"""

import os 
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv 
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate 
from langchain_core.documents import Document
from langchain_classic.schema import AIMessage 
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.chains import LLMChain
load_dotenv()
from helper_functions import *
logging.basicConfig(level= logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
embedding = OpenAIEmbeddings()


def extract_text(item):
    """
    Extract text content from either a string or an AIMessage object
    """
    if isinstance(item, AIMessage):
        return item.content
    return item


def embed_text(texts: list[str])->list[list[float]]:
    
    logging.info(f"Embedding {len(texts)} texts")
    return embedding.embed_documents([extract_text(text) for text in texts])



def perform_clustering(embedding: np.ndarray, n_cluster: int = 10)->np.ndarray:
    """
    Performent clustering on embedding using Gaussian Mixture model.
    """
    logging.info(f"Performing cluster with {n_cluster} cluster")
    gm = GaussianMixture(n_components= n_cluster, random_state=42)
    return gm.fit_predict(embedding)


def summary_text(texts : list[str]) -> str:
    """
    Summary list of text using OpenAI
    """
    prompt = ChatPromptTemplate.from_template(
        "Summaries the following text concisely:\n\n{text}"
    )
    chain = prompt | llm
    input_data = {"text": texts}
    return chain.invoke(input_data)
def visualize_cluster(embeddings: np.ndarray, labels: np.ndarray, level: int):
    """
    Visualize cluster using PCA
    """

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization - Level {level}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
"""
===================================================
RAGTOR core Function
===================================================
"""

def build_ragtor_tree(texts: list[str], max_levels: int = 3) ->dict[int, pd.DataFrame]:
    """
    Build the Ragtor tree structure with level metadata and parent child relationships
    """

    results = {}
    current_texts = [extract_text(text) for text in texts]
    current_metadata = [{"level":0, "origin": "original", "parent_id": None} for _ in texts]



    for level in range(1, max_levels+1):
        logging.info(f"Process level: {level}")

        embedding = embed_text(current_texts)
        n_clusters = min(10, len(current_texts)//2)
        cluster_lables = perform_clustering(embedding, n_clusters)

        df = pd.DataFrame(
            {
                'text':current_texts,
                'embedding': embedding,
                'cluster': cluster_lables,
                'metadata': current_metadata
            }
        )

        results[level-1] = df

        summaries = []
        new_metadata = []

        for cluster in df["cluster"].unique():
            cluster_docs = df[df['cluster'] == cluster]
            cluster_texts = cluster_docs['text'].tolist()
            cluster_metadata = cluster_docs['metadata'].tolist()
            summary = summary_text(cluster_texts)
            summaries.append(summary)
            new_metadata.append({
                'level':level,
                'origin': f'summary_of_cluster_{cluster}_level_{level-1}',
                'child_id': [meta.get('id') for meta in cluster_metadata],
                'id':f'summary_{level}_{cluster}'
            })
        current_texts = summaries
        current_metadata = new_metadata

        if len(current_texts)<=1:
            results[level] = pd.DataFrame(
                {
                    'text': current_texts,
                    'embedding': embed_text(current_texts),
                    'cluster':[0],
                    'metadata':current_metadata
                }
            )
            logging.info(f"Stopping at level {level} as we have only one summary")
            break 

    return results


"""
=====================================
Vectorestore function
=====================================
"""
def build_vectorestore(tree_results : dict[int, pd.DataFrame]) -> FAISS:
    """
    Build a FAISS vectorestore from all text in the RARTOR tree
    """
    all_texts =[]
    all_embeddings = []
    all_metadatas = []

    for level, df in tree_results.items():
        all_texts.extend([str(text) for text in df['text'].tolist()])
        all_embeddings.extend([embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in df['embedding'].tolist()])
        all_metadatas.extend(df['metadata'].tolist())

    logging.info(f"Building vecterstore with {len(all_texts)} texts")

    documents = [Document(page_content = str(text), metadata= metadata)
                for text, metadata in zip(all_texts, all_metadatas)]


    return FAISS.from_documents(documents,embedding)

"""
=====================================
define tree traversal retrieval
=====================================
"""
def tree_traversal_retrieval(query: str, vectorestor: FAISS, k: int = 3)-> list[Document]:
    """
    Perform tree travesal retrieval
    """
    query_embedding = embedding.embed_query(query)
    def retrievel_level(level: int, parent_ids: list[str] = None)->list[Document]:
        if parent_ids:
            docs = vectorestor.similarity_search_by_vector_with_relevance_scores(
                query_embedding,
                k = k,
                filter = lambda meta: meta['level'] == level and meta[id] in parent_ids
            )
        else:
            docs = vectorestor.similariry.search_ny_vector_with_relevance_scores(
                query_embedding,
                k = k,
                filter = lambda meta : meta['level'] == level
            )

        if not docs or level ==0:
            return docs

        child_ids = [doc.metadata.get('child_ids', []) for doc, _ in docs]
        # Flatten the list
        child_ids = [item for sublist in child_ids for  item in sublist]

        child_docs = retrievel_level(level-1, child_ids)
        return docs + child_docs

    max_level = max(doc.metadata['level'] for doc in vectorestor.docstore.values())
    return retrievel_level(max_level)


"""
===========================
create retriever
===========================
"""
def create_retriever(vectorestore:FAISS) -> ContextualCompressionRetriever:
    """
    Create retriever with contextual compression
    """
    logging.info("Create retriever compression retriever")
    base_retriever = vectorestore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        Given folowing context and question, extract only relevant information for answering the question:

        Context: {context}
        Question: {question}


        Relevant information: 
        """
    )
    extractor = LLMChainExtractor.from_llm(llm, prompt =prompt)
    return ContextualCompressionRetriever(
        base_compressor = extractor,
        base_retriever = base_retriever
    )



"""
===========================
Define hierarchical retrieval
===========================
"""

def hierarchical_retrieval(query: str, retriever: ContextualCompressionRetriever , maxlevel: int = 3) ->list[Document]:
    """
    Perform hierarchical retrieval starting from the highest level, Handling potential None value
    """

    all_relevanved_docs =[]
    for level in range(maxlevel, -1, -1):

        # retrieval document from current level
        docs_level = retriever.invoke(
            query,
            filter= lambda meta: meta['level'] == level
        )
        all_relevanved_docs.extend(docs_level)
        if docs_level and level>0:
        # if found the documents, retrieve their children from the next level down
            child_ids = [doc.metadata.get("child_ids", []) for doc in docs_level]
            child_ids = [item for sublist in child_ids for item in sublist if item is not None]


            if child_ids:  # Only modify query if there are valid child IDs
                child_query = f" AND id:({' OR '.join(str(id) for id in child_ids)})"
                query += child_query

    return all_relevanved_docs


"""
===========================
RAGTOR query process
===========================
"""

def raftor_query(query: str, retriever: ContextualCompressionRetriever, maxlevel: int)->  dict[str, Any]:
    """
    Process query using the RAGTOR system with hierarchical retrieval
    """
    logging.info(f"Process query: {query}......")
    relevance_docs = hierarchical_retrieval(query, retriever, maxlevel)

    prompt = ChatPromptTemplate.from_template(
        """
        Give the folowing context, please answer the question:

        Context:{context}
        Question:{question}

        Answer:
        """
    )
    chain = prompt | llm

    detail_docs =[]
    for i, doc in enumerate(relevance_docs,1):
        detail_docs.append(
            {
                "index": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "level": doc.metadata.get('level', 'Unknown'),
                "similarity_score": doc.metadata.get('score', 'N/A')
            }
        )

      
    context = '\n\n'.join([doc.page_content for doc in relevance_docs])

    answer = chain.invoke({"context": context, "question": query}).content

    logging.info("Query processing completed")

    result ={
        "query": query,
        "retrieved_documents": detail_docs,
        "num_docs_retrieved": len(relevance_docs),
        "context_used": context,
        "answer": answer,
        "model_used": llm.model_name
    }
    return result




def print_query_details(result: dict[str, Any]):
    """Print detailed information about the query process, including tree level metadata."""
    print(f"Query: {result['query']}")
    print(f"\nNumber of documents retrieved: {result['num_docs_retrieved']}")
    print(f"\nRetrieved Documents:")
    for doc in result['retrieved_documents']:
        print(f"  Document {doc['index']}:")
        print(f"    Content: {doc['content'][:100]}...")  # Show first 100 characters
        print(f"    Similarity Score: {doc['similarity_score']}")
        print(f"    Tree Level: {doc['metadata'].get('level', 'Unknown')}")
        print(f"    Origin: {doc['metadata'].get('origin', 'Unknown')}")
        if 'child_docs' in doc['metadata']:
            print(f"    Number of Child Documents: {len(doc['metadata']['child_docs'])}")
        print()
    
    print(f"\nContext used for answer generation:")
    print(result['context_used'])
    
    print(f"\nGenerated Answer:")
    print(result['answer'])
    
    print(f"\nModel Used: {result['model_used']}")


if __name__ == "__main__":
    path = "data/Understanding_Climate_Change.pdf"
    loader = PyPDFLoader(path)
    documents = loader.load()

    texts = [doc.page_content for doc in documents]

    tree_result = build_ragtor_tree(texts)
    vectorstore = build_vectorestore(tree_result)
    retriever = create_retriever(vectorstore)

    max_levels = 3
    query = "What is the greehouse effect?"
    results = raftor_query(query, retriever, max_levels)
    print_query_details(results)
