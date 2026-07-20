"""
Implement three query transformation techniques in enhance the retriever process

1 query rewriting: Reformulates query to be more specific and detail -> improve likelihood to retrieving relevant information, use gpt 4 to write structure
2 step-back prompting: Genearate broader queries for better context retrieval -> more general queries can help retriever relevant  background information
3 sub-query decomposition: break down complexed queries into simple sub queries 


Query transformtion techniques addess retrieving the most relevant information by reformulation queres to match relevant document or to retrieve more 
comprehensive information

Example use case:

Query rewriting:  expand this specific aspects like temperature changes and biodiversity
step_back prompting:  generlizes it to : what are the general effect if climate change?
sub-query decomposition: break down it into question about biodiversity about  biodiversity, oceans, weather patterns and terrestial enviroments 

"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import  PromptTemplate
# rewrite query 
def rewrite_query(query):
    """
    Using model llm to rewite query from original query
     arg:
        query (str): original query
    return:
        rewritten (str): query
    """
    re_writer_lmm = ChatOpenAI(model = "gpt-4.0-mini", temperature= 0, max_token = 4000)

    query_write_template = """
        You are an AI assistant tasked with reformulation user queries to improve context retrieval in a RAG system.
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

        Original query: {query},
        Rewrite query: 
    """

    query_rewrite = query_write_template | re_writer_lmm
    response =  query_rewrite.invoke(query)
    return response.content



"""step back prompting: """
def step_back_prompting(query):
    """
    Generater broading queries for better context retrieval
    arg:
        query (str): original query
    return:
        response str: The step-back query
    
    """
    step_back_llm = ChatOpenAI(model = "gpt-4o", temperature  = 0, mex_tokens = 4000)
    query_stepback_template = """
        You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
        Given the original query, generate a step-back query that is more general and can help retrieve relevant background information
        Original query: {query},
        Step-back query:
    """
    step_back = query_stepback_template | step_back_llm
    response = step_back.invoke(query)
    return response.content


""" Sub-query Decomposition"""
def sub_query_prompting(query):
    """
    Breaking complex queries into simpler sub-queries
    arg:
    query str: original query
    return:
    response str: sub-query
    """
    sub_query_llm = ChatOpenAI(model = "gpt-4o", temperature = 0, max_tokens = 4000)
    sub_query_prompting = """
        You are an AI assistant tasked with breaking down complexition query into simpler sub-quiries for a RAG system.
        Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, could provide comprehensive response to the original query.
        Original query: {query},
        Example: What the impact of climate change on the environment?
        Sub-queries:
        1. What are impacts of climate change on biodiversity?
        2. How does climates changes affect the ocean?
        3. What are imparts of climates change on the agriculture?
        4. What are imparts of climates change on the human health?
    """
    sub_query = sub_query_prompting | sub_query_llm
    response = sub_query.invoke(query)
    return response.content
if __name__ =="__main__":
    original_query = "What are the impacts of climate change on the environment?"
    sub_queries = sub_query_prompting(original_query)

    print("\nSub-queries:")
    for i, sub_query in enumerate(sub_queries, 1):
        print(sub_query)


