"""
Retriever segment extraction(RSE)
this is method of reconstructing multi-chunk segments  of contiguous text out of retriever chunks.
This steps occus after vectorstore ( and optional reranking) but before presenting the retriever contest to the LLM.
This method ensure that nearby chunks are presented to the LLM in the other they appear in the original document. It also 
add in chunks that are not marked as relevant, but are sandwitched between  hightly relevant chunks, further improving  the context provided to the LLM.

Method detail:
Ducument chunking:
Standard documents chunking can be used. The only splecial requirement here is that document are chunk with no overlap. This allow us to reconstruct sections of
the document (eg segnent)by concatenating chunk.



RSE Optimization:
After standard chunk retriever process is  completed, which idea included a reranking step, the RSE can be beign.
1. Combine the absolute relevance value and the relevance rank.
2. we suctract a contant threshold value (let say 0.2) for each chunk'value
 By calculating chunk values this way we can define segment value as just the sum of the chunk values.


Combine absolute                subtract constant threshold             find maximum                                 construct full segment text by retrieving
relevance score with     ->             value                   ->      sum subarray          ->                      from chunk store
relevant rank
                                                                               |                                             |
                                                                               |                                             |
                                                                               +---------------------------------------------+
                                                                                             Repeat until no more segments
                                                                                      cross the minimum score threshold
"""
import os
import cohere
import numpy as np

from dotenv import load_dotenv
from scipy.stats import beta
import matplotlib.pyplot as plt 

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# define some parameters and constraints for the optimization
irrelevant_chunk_penalty = 0.2 # empirically, something around 0.2 works well; lower values bias towards longer segments
max_length = 20
overall_max_length = 30
minimum_value = 0.7

def split_into_document(text: str, chunk_size: int):
    """
    Splitter a given document of specified size using RecursiveCharacterTextSplitter
    
    ARG:
    text (str): the input text to be split into chunk
    chunk_size (int, Optional): the Maximun size of each chunk, default is 800.
    Return:
    list[str]:A list of text chunk

    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = 0,
        length_function = len
    )
    texts = splitter.create_documents([text])
    chunks = [text.page_content for text in texts]
    return chunks 


def transform(x: float):
    """
    Transformation function to map the absolute relevance value that is more uniformly distributed between 0 and 1.
    The relevant given by corehere reranker tend to be very close to 0 or 1.
    This beta function used here help to spread out the values more uniformly

    ARG:
    x(float): the absolute relevant value return by Cohere reranker
    Return:
    float: the tranformed rerank value
    """
    a, b = 0.4, 0.4
    return beta.cdf(x, a,b)


def rerank_chunks(query: str, chunks: list[str]):
    """
    Use Corhere Reranking API to rerank the search results
    Arg:
    query(str): the search query
    chunks(list):List of chunk to be reranked

    reuturn:
    similarity_scores(list): List of similarity score for each chunk
    chunk_values(list): List of relevant values (fusion of ranking and similarity) of each chunk
    """


    model = "rerank-english-v3.0"
    client = cohere.Client(api_key= os.environ["CO_API_KEY"])
    decay_rate = 30

    rerank_result = client.rerank(model = model, query = query, documents = chunks)
    results = rerank_result.results
    rerank_indices = [result.index for result in results]
    rerank_similate_scores = [result.relevance_score for result in results]

    # convert back to other of original documents and calculate chunks values
    similarity_scores = [0]*len(chunks)
    chunk_values = [0]*len(chunks)

    for i, index in enumerate(rerank_indices):
        absolute_relevance = transform(rerank_similate_scores[i])
        similarity_scores[index] = absolute_relevance
        chunk_values[index] = np.exp(-i/decay_rate)*absolute_relevance # decay the relevance value based on the rank

    return similarity_scores, chunk_values


def plot_relevance_scores(chunk_values: list[float], start_index: int = None, end_index: int = None) -> None:
    """
    Visualize of each chunks in the document to the search query    
    Args:
        chunk_values (list): List of relevance values for each chunk
        start_index (int): Start index of the chunks to be plotted
        end_index (int): End index of the chunks to be plotted

    Returns:
        None

    Plots:
        Scatter plot of the relevance scores of each chunk in the document to the search query
    """

    plt.figure(figsize=(12, 5))
    plt.title(f"Similarity of each chunk in document  to the search query")
    plt.ylim(0,1)
    plt.xlabel("Chunk index")
    plt.ylabel("Query-chunk similarity")
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(chunk_values)
    plt.scatter(range(start_index, end_index), chunk_values[start_index:end_index])
    plt.show()


def get_best_segments(relevance_values: list, max_length: int, overall_max_length:int, minimum_value: float):
    """
    This function takes a chunks relevance value and then runs an optimization algorithm to find  the best segments.
    In more technical terms, it show constrained version of the maximum sum subarray problem.
    Note: this is similified implementation intended for demontration purpose. 
    A more sophisticated implementation would be needed for production use and is available in the dsRAG library.

    Args:
        relevance_values (list): a list of relevance values for each chunk of a document
        max_length (int): the maximum length of a single segment (measured in number of chunks)
        overall_max_length (int): the maximum length of all segments (measured in number of chunks)
        minimum_value (float): the minimum value that a segment must have to be considered

    Returns:
        best_segments (list): a list of tuples (start, end) that represent the indices of the best segments (the end index is non-inclusive) in the document
         scores (list): a list of the scores for each of the best segments
    """
    best_segments = []
    scores = []
    total_length = 0
    while total_length < overall_max_length:
        # find the best remaining segment
        best_segment = None
        best_value = -1000
        for start in range(len(relevance_values)):
            # skip over negative value starting points
            if relevance_values[start] < 0:
                continue
            for end in range(start+1, min(start+max_length+1, len(relevance_values)+1)):
                # skip over negative value ending points
                if relevance_values[end-1] < 0:
                    continue
                # check if this segment overlaps with any of the best segments and skip if it does
                if any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments):
                    continue
                # check if this segment would push us over the overall max length and skip if it would
                if total_length + end - start > overall_max_length:
                    continue
                
                # define segment value as the sum of the relevance values of its chunks
                segment_value = sum(relevance_values[start:end])
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)
        
        # if we didn't find a valid segment then we're done
        if best_segment is None or best_value < minimum_value:
            break

        # otherwise, add the segment to the list of best segments
        best_segments.append(best_segment)
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]
    
    return best_segments, scores



if __name__ == "__main__":
    PATH = "data/nike_2023_annual_report.txt"
    with open(PATH, 'r') as file:
        text = file.read()
    chunks = split_into_document(text, chunk_size= 800)
    print (f"Split the document into {len(chunks)} chunks")


    query = "Nike consolidated financial statements"

    similarity_scores, chunk_values = rerank_chunks(query, chunks)
    # plot_relevance_scores(chunk_values)


    relevance_values = [v - irrelevant_chunk_penalty for v in chunk_values] 

    # run the optimization
    best_segments, scores = get_best_segments(relevance_values, max_length, overall_max_length, minimum_value)

    # print results
    print ("Best segment indices")
    print (best_segments) # indices of the best segments, with the end index non-inclusive
    print ()
    print ("Best segment scores")
    print (scores)
    print ()