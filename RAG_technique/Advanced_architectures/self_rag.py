"""
Overview:
Self-Rag is an advanced algorithm that combines the power of the retrieval-based and generation-based approaches in NLP.
It dynamically dicides Whether to use retriever information and how to best utilize it in generating responses, aim to produce more accurate,
relevant and useful output.

Motivation:
Tradditional quesion-answering system often struggle with bandlancing the use of retrieved information and generate of new content. 
Some systems might rely on heavily on retrieved data, leading to responses that lack fexibility, while orthers might generate without suffcient grounding in factual information.
Self-RAG addresses these issues by implementing a multi-step process that carefully evaluates and necessity and relevant of retrieved information, and assesses the quality of generator responses.



Methob Details:
1. Retrival Decision: The algorithm first decides if retrieval is necessary for given query. This step prevents unnecessary for query that can be answered direcly.
2. Document Retrieval: If retrieval is deemed necessary, the algorithm fetches the top-k most similar document for vectorstore.
3. Relevant Evaluation: Each retrieved document is eveluated for its relevant to the query. This step filter out irrelevent information, ensuring that only pertinent context is used for generation.
4. Response generation: The algorithm generate response using relevant contexts. If no relevant contexts are found, it generate a response without retrieval.
5. Support Assessment: Each generated response is evaluated for determine how well it is supported by the context. This step help in identifying responses that are grounded in the provided information,
6. Utility Evaluation: The utility of each response is rated, considery how well it addresses the ordinary query.
7. Response Selection: The final step involves selecting the best response base on the support assessment and utility evaluation

"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_spliters import RecursiveCharacterTextSplitter


from helper_functions import *

"""
======================================================================================================
 define the prompt template : Retrival, Relevant, Generate Response, Support Response, Utility Response
======================================================================================================
"""
class RetrivalResponse(BaseModel):
    response : str = Field(..., title="Determines if retrieval is necessary", description = "Output only 'Yes' or 'No'.")
class RelevantResponse(BaseModel):
    response : str = Field(..., title = "Determinies if context is relevant", description = "Outout only 'Relevant' or 'Irrelevant'.")
class GenerateResponse(BaseModel):
    response : str = Field(..., title = "Generate response", description = "The generated response.")
class SupportResponse(BaseModel):
    response : str = Field(..., title = "Determinies if response is supported", description = "Output 'Fully supported', 'Partially supported',or 'No support'.")
class UtilityResponse(BaseModel):
    response : str = Field(..., title = "Utility rating", description = "Rate the utility of the response from 1 to 5.")



retrival_prompt = PromptTemplate(
    input_variables = ["query"],
    template = "Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
)
relevant_prompt = PromptTemplate(
    input_variables = ["query", "context"],
    template = "Given the query '{query}' and the context '{context}', detemine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)
generate_prompt = PromptTemplate(
    input_variables = ["query", "context"],
    template = "Given the query '{query}' and the context '{context}', generate a response."
)
support_prompt = PromptTemplate(
    input_variables = ["response", "context"],
    template = "Given the response '{response}' ans the context '{context}', determine if the response is supported by context. Output 'Fully supported', 'Partially supported',or 'No support'."
)
utility_prompt = PromptTemplate(
    input_variables = ["query", "response"],
    template = "Give the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)



llm = ChatOpenAI(model = "GPT-4o-mini", max_tokens = 1000, temperature = 0 )
retrieval_chain = retrival_prompt | llm.with_structured_output(RetrivalResponse)
relevant_chain = relevant_prompt | llm.with_structured_output(RelevantResponse)
generate_chain = generate_prompt | llm.with_structured_output(GenerateResponse)
support_chain = support_prompt | llm.with_structured_output(SupportResponse)
utility_chain = utility_prompt | llm.with_structured_output(UtilityResponse)
"""
======================================================================================================
Defining the self RAG logic flow
======================================================================================================
"""


def encode_pdf(path, chunk_size = 1000, chunk_ovelap = 2000):
    loader = PyPDFLoader(path)
    docs = loader.load()
    slipter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_ovelap= chunk_ovelap,
        length_function = len
    )
    embedding =OpenAIEmbeddings()

    docs_slipter = slipter.split_documents(docs)
    clean_docs = replace_t_space(docs_slipter)
    vectorstore = FAISS.from_documents(clean_docs, embedding)
    return vectorstore
def self_rag(query: str, vectorstore: FAISS, top_k : int = 3):
    """
    Process self_rag 
    ARG:
    query (str): input of user query
    vectorstore (FAISS): vectorstore was encode from openAI model and using FAISS
    return:
    best_response of user query  
    """
    #step1: determine if retrieval is necessary
    input_data = {"query": query}
    retrieval_decision = retrieval_chain.invoke(input_data).response.strip().lower()
    print(f"Retrival dicision: {retrieval_decision}")
    #step2: retrieval relevant document
    if retrieval_decision == "Yes":
        relevant_docs = vectorstore.similar_search(query, k = top_k)
        contexts = [doc for doc in relevant_docs]
        print(f"Retrieval {len(contexts)} documents")

        relevant_contexts = []
        #step3: Evaluate relevance of retrieval document
        for i, context in enumerate(contexts):
            input_data = {"query":query, "context": context}
            relevance = relevant_chain.invoke(input_data).response.strip().lower()
            print(f"Document {i+1} relevance: {relevance}")
            if relevance == "Relevant":
                relevant_contexts.append(context)

        #if no relevance document, generate without retrieval
        if not relevant_contexts:
            print("No relevant context found. Generate without retrieval")
            input_data = {"query": query, "context": "No relevant context found."}
            return generate_chain.invoke(input_data).response

          
        #step4:Generate response using the relevant context
        responses = []
        for i, context in enumerate(relevant_contexts):
            print("Generate the reponse with relevant context")
            input_data = {"query": query, "context": context}
            response = generate_chain.invoke(input_data).response

            #step5: assest support
            print(f"Assessing support for the response: {i+1}")
            input_data = {"query": query, "context": context}
            support = support_chain.invoke(input_data).response
            print(f"Support assestment: {support}")
            #step6: evaluate utility
            print(f"Evaluation utility: {i+1}")
            input_data = {"qeury": query, "response": response}
            utility = int(utility_chain.invoke(input_data).response.strip().lower())
            responses.append((response, support, utility))
            
        # Select best response base on support and query
        print("Select the best response")
        best_response = max(responses, key= lambda x: (x[1] == "Fully supported", x[2]))
        print(f"Best response support: {best_response[1]}, utitlity: {best_response[2]}")
        return best_response[0]
    
    # generate without retrieval
    else:
        print("Generating without retrieval...")
        input_data = {"query":query, "context":"No retrieval necessary"}
        return generate_chain.invoke(input_data).response


if __name__ =="__main__":
    path = "data/Understanding_Climate_Change.pdf"
    vectorstore = encode_pdf(path)

    query = "What is the impact of climate change on the environment?"
    response = self_rag(query, vectorstore)
    print("\n\nFinal Answer:\n")
    print(response)