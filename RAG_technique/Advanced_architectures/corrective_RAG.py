"""
Corrective RAG process is an advanced information retrieval and response generationg system.
It extends the standards RAG approach by dynamically evaluation and connecting the retrieval process, combine the power of vectore databases, web search, LLM to provide the accurate and context-aware
response to user queries.


Motivation:
While tradditional RAG system have improved imformation retrieval and response generation.
they can still fall short when then retrieved information is irrelevant and outdated.

The Correct RAG process addresses these limitation by:
1. Levergratin pre-existing knowledge base.
2. Evaluation the relevance of retrieval information.
3. Dynamically searching  the web when necessary.
4. Refining and combining knowledge from multiple search
5. Generate human-like response on the most appropriate knowledge


Key Components
1. FAISS index: A vectore database for efficient similarity search for pre-exiciting knowledge.
2. Retrieval Evaluation: Assess the relevant of retrieval of retrieval document to the query.
3. Knowledge refinement: Extract key information from documents when necessary.
4. Web serch query rewrite: Optimize queries for the web searches when local knowledge is insufficient
5. Response generator: Create human-like responses base on the accumulated knowledge


Method Details:
1. Document Retrieval:
    Performent similar search in the Faiss index to find relevant documents
    Retrieval with top k =3 (default)
2. Document Evaluation:
    Caculate relevant search for each retrieved document
    Determines the best couse of action base on the highest relevance score
3. Corrective knowledge Acquisition:
    If high relevance (score>0.7): Use the most relevant document as-is.
    If low relevance (score<0.3): Correctly by performing a wed search with a retrieval query
    If ambiguous (0.3<=score<=0.7): Correct by combining the most relevant document with web search result
4. Adaptive knowledge processing:
    For Web search results: Refine the knowledge to extract key point
    For ambiguous cases: Combined raw document content with refined web search results 
5. Response Generation:
    Uses a language model to generate a human-like response base on the query and acquired knowledge
    Include source information in the response for transparency
"""

import os 
import sys
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorestores import FAISS
from langchain_text_splitters import RecusiveCharacterTextSplitter 
from langchain_core.prompts import PromptTemplate

from helper_functions import *
load_dotenv()


"""
============================================
Define retrieval documents function for vectorstore, search tool, LLM model
============================================
"""
llm = ChatOpenAI(model = "gpt-4o-mini", max_tokens =1000, temperature = 0 )
search = DuckDuckGoSearchResults()

def encode_PDF(path, chunk_size = 1000, chunk_overlap = 200):
    """
    Create vector store from FAISS using Open AI to embeddding
    
    ARGs:
    path (str): path to file PDF
    chunk_size (int): number tokens of each chunk when split document
    chunk_overlap (int): the numbers of token repeat from pre-chunk in current chunk
    
    Return:
    vectorestore (FAISS)
    """
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecusiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    embedding = OpenAIEmbeddings()

    docs_splitters = splitter.split_documents(docs)
    clean_docs = replace_t_space(docs_splitters)
    vectorstore = FAISS.from_documents(clean_docs, embedding)

    return vectorstore


"""
============================================
Define retrieval evaluation, knowledge refinement and query rewrite llm chains 
============================================
"""
# retrieval evaluator
class retrieval_evaluation(BaseModel):
    response :float = Field(..., description = "The relevance score of ducument to the query. The score should be between 0 and 1.")

# knowledge refinement
class KnowledgeRefinement(BaseModel):
    response : str = Field(..., description = "The document tp extract key information from.")

# Web Search query rewrite
class WebSearchRewrite(BaseModel):
    response : str = Field(..., description = "The query to rewrite")

def retrieval_evaluator(query: str, document: str) ->float:
    """
    Evalution relevant between query and document

    ARGd:
        query (str): Input user query
        document (str): the knowledge base
    return:
        the score after evaluate (0-1)
    """
    prompt = PromptTemplate(
        input_variables = ["query", "document"],
        template = "On the scale from 0 to 1, How relevant is the following document to the query? Query: {query}\n Document: {document}\n Relevance score:"
    )
    chain = prompt | llm.with_structered_output(retrieval_evaluation)
    input_data = {"query": query, "document": document}
    results = chain.invoke(input_data).relevance_score
    return results

def knowledge_refinement(document: str) -> list[str]:
    """
    Extract key infotmation from document in bullet point

    ARGs:
        document (str): the knowledge base want 
    return:
        list[str]: list key informtion from document
    """
    prompt = PromptTemplate(
        input_variables = ["document"],
        template = "Extract the key information from the following document in bullet point:\n{document}\nKey Point:"
    )
    chain = prompt | llm.with_structured_output(KnowledgeRefinement)

    input_data = {"document": document}
    results = chain.invoke(input_data).key_points
    return [point.strip() for point in results.split('\n') if point.strip()]


def rewrite_query(query:str)->str:
    """
    Rewrite the query user to suitable for the web search
    """
    prompt = PromptTemplate(
        input_variables =["query"],
        template = "Rewrite the following query to make it more suitable for a web search:\n{query}\nRewriten query:"
    )
    input_data = {"query": query}

    chain = prompt | llm.with_structured_output(WebSearchRewrite)    
    retults = chain.invoke(input_data).query.strip()
    return retults

"""
============================================
Helper fuction to parse search results 
============================================
"""
def parse_search_results(results_string: str)-> list[tuple[str, str]]:
    """
    Parse a Json string of search result into list of tille-link tuple

    ARGs:
        results_string (str): A Json fromat string  containing search results
    
    Return:
        list[tuple[str, str]]: A list of tuple string, where each tuple contains the title and link of a search result.
                                if parse fails, the empty list is returned
    """

    try:
        # Attempt to parse the Json string
        results = json.loads(results_string)
        # Extract and return the tittle and link from each results
        return [(result.get('title', 'untitled'), result.get('link','')) for result in results]
    except json.JSONDecodeError:
        print("Error parsing search results. Returning empty list.")
        return []


"""
============================================
Define dub function for CRAG process : Retrieval documents, evaluation docuemnts, perform web search, generate response 
============================================
"""
def retrieval_documents(query: str, faiss_index: FAISS, k : int = 3) -> list[str]:
    """
    Retrieval Document base on query using FAISS index.

    Args:
        query(str): the query string to search for
        faiss_index (Faiss): The Faiss index user for similarity search
        k (int): The number of top documents to retrieval (default: 3)
    
    Return:
        list(str): A list of retrieval document content
    """
    docs = faiss_index.similarity_search(query, k = k)
    return [doc.page_content for doc in docs ]

def evaluate_documents(query: str, documents: list[str]) ->list[float]:
    """
    Evaluation documents base on query

    ARGs:
        query(str): the query string to search for
        documents list(str): a list of documents content to evaluate

    Returns:
        list[float]: list relevance score after evaluation from 0-1 
    """
    return [retrieval_evaluation(query, document) for document in documents]

def perform_web_search(query: str) ->tuple[list[str],list[tuple[str, str]]]:
    """
    Performemt web searching base on query

    ARGs:
        query (str): the query string to search for
    return:
        tuple[list[str], list[tuple[str, str]]]
            + a list of refined knowledge obtained from the web search
            + A list of tuple containing titles and links of the process
    """
    rewritten_query = rewrite_query(query)
    web_results = search.run(rewritten_query)
    web_knowledge = knowledge_refinement(web_results)
    source =parse_search_results(web_results)
    return web_knowledge, source

def generate_response(query: str, knowledge: str, source: list[tuple[str, str]])-> str:
    """
    Generate response to a query using knowledge and response

    ARGs:
        query (str): the query string to search for
        knowledge (str): the refine knowledge to use in the response 
        source list[tuple[str, str]]: A list of tuples containing titles and links on the sources
    return:
        str: the Generate reponse
    """
    response_prompts = PromptTemplate(
        input_variables = ["query", "knowledge", "source"],
        prompt = "Base on the following knowledge, answer the query. Inculde the source with their links(if available) at the end of your answer:\nQuery: {query}\nKnowledge{knowledge}\nSource: {source}\nAnswer:"
    )
    chain = response_prompts | llm
    input_variables = {
        "query": query,
        "knowledge": knowledge,
        "source": '\n'.join([f"{title}:{link}" if link else title for title, link in source])
    }

    response = chain.invoke(input_variables).content
    return response


"""    
============================================
Crag process 
============================================
"""

def crag_process(query: str, faiss_index: FAISS)-> str:
    """
    Process a query by retrieving, evaluation, using documents or performent web search to gernerate response

    Args:
        query (str): the query string to process
        faiss_index (FAISS): the FAISS index user for document retrieval
    return: 
        str: the generated base on the query
    """
    print(f"Processing query: {query}")

    retrieved_docs = retrieval_documents(query, faiss_index)
    eval_scores = evaluate_documents(query, retrieved_docs)

    print(f"\nRetrieved {len(retrieved_docs)} documents")
    print(f"Evaluation scores: {eval_scores}")


    max_score = max(eval_scores)

    sources = []
    if max_score>0.7:
        print("\nAction: Correct - Using retrieved document")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        final_knowledge = best_doc
        sources.append(("Retrieved document", ""))
    elif max_score < 0.3:
        print("\nAction: Incorrect - Performing web search")
        final_knowledge, sources = perform_web_search(query)
    else:
        print("\nAction: Ambiguous - Combining retrieved document and web search")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        # Refine the retrieved knowledge
        retrieved_knowledge = knowledge_refinement(best_doc)
        web_knowledge, web_sources = perform_web_search(query)
        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        sources = [("Retrieved document", "")] + web_sources

    print("\nFinal knowledge:")
    print(final_knowledge)
    
    print("\nSources:")
    for title, link in sources:
        print(f"{title}: {link}" if link else title)

    # Generate response
    print("\nGenerating response...")
    response = generate_response(query, final_knowledge, sources)

    print("\nResponse generated")
    return response


if __name__ == "__main__":
    path  = "data/Understanding_Climate_Change.pdf"
    vectorstore = encode_PDF(path)


    query = "What are the main causes of climate change?"
    result = crag_process(query, vectorstore)
    print(f"Query: {query}")
    print(f"Answer: {result}")


