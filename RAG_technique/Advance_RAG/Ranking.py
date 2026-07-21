"""
Reranking in RAG systems is to overcome limitations of initial retrieval methods, which often rely on simpler similarity metrics. 
Reranking allows for more sophisticated relevance assessment, 
taking into account nuanced relationships between queries and documents that might be missed by traditional retrieval techniques. 
This process aims to enhance the overall performance of RAG systems by ensuring that the most relevant information is used in the generation phase.


key compenents:
1. Initial Retriever: often a vector store using embedding-based similary search
2. Reranking Model: This can be either:
    + A LLm for scoring relevance
    + A cross- encoder model specifically trained for relevance assessment
3. Scoring Mechanism: A method to assign relevance scores to documents
4. Sorting and Selection Logic: To reorder documents based on new scores


Method Details:
The reranking process generally follows these steps:
1.Initial Retrieval: Fetch an initial set of potentially relevant documents.
2.Pair Creation: Form query-document pairs for each retrieved document
3.Scoring:
    +LLM Method: Use prompts to ask the LLM to rate document relevance.
    +Cross-Encoder Method: Feed query-document pairs directly into the model.
4.Score Interpretation: Parse and normalize the relevance scores.
5.Reordering: Sort documents based on their new relevance scores
6.Selection: Choose the top K documents from the reordered list.

"""


import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Feild

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbedding
from sentence_transformers import CrossEncoder

from helper_functions import *

load_dotenv()


class RatingScore(BaseModel):
    relevance_score : float = Field(..., description = "The relevance score of a document to a query")

def rerank_documents(query: str, docs: list[Document], top_k: int = 3)-> list[Document]:
    prompt_template = PromptTemplate(
        input_variables = ["query", "doc"],
        template = """On the scale of 1-10, rate the relevance of following document to the query.
        Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:
        """
    )
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0, max_tokens = 4000)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))
    
    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_k]]


# create a custom retrieve base on our Rerank
class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore :Any = Field(description = "Vector store for inital retrival")
    class config:
        arbitrary_types_allowed = True
    def get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(query, initial_docs, top_n=num_docs)
    



cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort documents by score
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top reranked documents
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")
    

def process_method_1(query):
    path = "data/Understanding_Climate_Change.pdf"
    vectorstore = encode_pdf(path)
    custom_retriever = CustomRetriever(vectorstore=vectorstore)

    # Create an LLM for answering questions
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    # Create the RetrievalQA chain with the custom retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    print(f"\nQuestion: {query}")
    print(f"Answer: {result['result']}")
    print("\nRelevant source documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:200] + "...")



def process_method_2(query):
    path = "data/Understanding_Climate_Change.pdf"
    vectorstore = encode_pdf(path)
    cross_encoder_retriever = CrossEncoderRetriever(
        vectorstore=vectorstore,
        cross_encoder=cross_encoder,
        k=10,  # Retrieve 10 documents initially
        rerank_top_k=5  # Return top 5 after reranking
    )

    # Set up the LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    # Create the RetrievalQA chain with the cross-encoder retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=cross_encoder_retriever,
        return_source_documents=True
    )

    # Example query
    query = "What are the impacts of climate change on biodiversity?"
    result = qa_chain({"query": query})

    print(f"\nQuestion: {query}")
    print(f"Answer: {result['result']}")
    print("\nRelevant source documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:200] + "...")

if __name__ == "__main__":

    query = "what is the capital of france?"
    # process_method_1(query)
    # process_method_2(query)










# chunks = [
#     "The capital of France is great.",
#     "The capital of France is huge.",
#     "The capital of France is beautiful.",
#     """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower. 
#     I really enjoyed all the cities in france, but its capital with the Eiffel Tower is my favorite city.""", 
#     "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
# ]
# docs = [Document(page_content=sentence) for sentence in chunks]


# def compare_rag_techniques(query: str, docs: List[Document] = docs) -> None:
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(docs, embeddings)

#     print("Comparison of Retrieval Techniques")
#     print("==================================")
#     print(f"Query: {query}\n")
    
#     print("Baseline Retrieval Result:")
#     baseline_docs = vectorstore.similarity_search(query, k=2)
#     for i, doc in enumerate(baseline_docs):
#         print(f"\nDocument {i+1}:")
#         print(doc.page_content)

#     print("\nAdvanced Retrieval Result:")
#     custom_retriever = CustomRetriever(vectorstore=vectorstore)
#     advanced_docs = custom_retriever.get_relevant_documents(query)
#     for i, doc in enumerate(advanced_docs):
#         print(f"\nDocument {i+1}:")
#         print(doc.page_content)


# query = "what is the capital of france?"
# compare_rag_techniques(query, docs)