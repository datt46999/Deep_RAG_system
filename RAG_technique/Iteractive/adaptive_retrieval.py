"""
Adaptive Retrieval Augumented Generation system
Implement advanced  retriever augumented generation approach that adapts its retrieval strategy base on type of query. By leveraging LLM in various stages, it aims to provide more accuracy, relevant, and context response to user queries



Key components:

1. Query classfier: Determines the types of query(Factual, Analytical, opinion or contextual)
2. Apdated retrieval stages: Four distinct strategies  tailored to different query type:
    + Factual strategy
    + Analytical strategy
    + Opinion strategy
    + Contextual strategy
3. LLM Intergration: LLm are used thoughtout the process to enchance  retrieval and ranking
4. Open AI Model: Generated final response usin the retrieval documents and context


Benefits of This Approach
Improved Accuracy: By tailoring the retrieval strategy to the query type, the system can provide more accurate and relevant information.

Flexibility: The system adapts to different types of queries, handling a wide range of user needs.

Context-Awareness: Especially for contextual queries, the system can incorporate user-specific information for more personalized responses.

Diverse Perspectives: For opinion-based queries, the system actively seeks out and presents multiple viewpoints.

Comprehensive Analysis: The analytical strategy ensures a thorough exploration of complex topics.

"""
import os
from typing import List, Any
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()

"""
Define query classifer class
"""
class categories_options(BaseModel):
    category :str = Field(description = "The category of the query, the options are: Factual, Analytical, Opion or Contextual", example = "Factual")


class relevant_score(BaseModel):
    score : float = Field(description="The relevance score of document  to the query", example =8.0)

class select_indices(BaseModel):
    indices : List[int] = Field(description = "Indices of select documents", example = [0, 1, 2, 3])

class SubQueries(BaseModel):
    sub_queries : List[str] = Field(description = "List of sub-queries for comprehensive analysis", example = ["What is the population of New York?", "What is the GDP of Viet Nam?"])
class queries_classifier:
    def __init__(self):
        self.llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0, max_tokens = 4000)
        self.prompt_system = PromptTemplate(
            input_variables = ["query"],
            template="Classifier the following query into one of these categories: Factial, Analytical, Opinion, Contextual. \nQuery: {query}\nCategory:"
        )
        self.chain = self.prompt_system | self.llm.with_structured_output(categories_options)
    def classify(self, query):
        print("Classifying query")
        return self.chain.invoke(query).category




"""
Define the Base Retriever class
"""

class BaseRetrievalStrategy:
    def __init__(self, texts):
        self.embedding = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size = 800, chunk_overlap = 0)
        self.documents = text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(self.documents, self.embedding)
        self.llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0, max_tokens = 400)
    def retrieve(self, query, k =4):
        return self.db.similarity_search(query, k)


"""
Define Factual strategy
"""

class FactualRetrieval(BaseRetrievalStrategy):
    def retrieve(self, query, k =4):
        print("Retrieving factual")    
        enchance_query_prompt = PromptTemplate(
            text_variables = ["query"],
            template = "Enhance this factual query for better information retrieval: {query}"
        )
        query_chain = enchance_query_prompt | self.llm
        enhance_query = query_chain.invoke(query).content
        print(f"Query after enhance: {enhance_query}")

        docs = self.db.similarity_search(enhance_query, k = k*2)

        # Use llm to ranking relevant of retrieval documents
        ranking_prompt = PromptTemplate(
            input_variables = ["query", "doc"],
            template = "On a scale of 1-10, how relevant this document to the query: {query} \n Document: {doc}\n Relevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(relevant_score)

        rankers_docs = []
        print("ranking docs: ")
        for doc in docs:
            input_data = {"query": enhance_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            rankers_docs.append((doc, score))

        # sort by relevance score and return top k
        rankers_docs.sort(key =lambda x: x[1], reverse=True )
        return [doc for doc, _ in relevant_score[:k]]


"""
Define Analytical reriever stragery
"""
class AnalyticalRetrieval(BaseRetrievalStrategy):
    def retrieve(self, query,k = 4):
        print("Analytical retrieval")
        # Use llm to generate sub-queries for comprehensive analysis
        sub_query_prompt = PromptTemplate(
            input_variables = ["query"],
            template = "Generate {k} sub-questions for: {query}"
        )
        sub_queries_chain = sub_query_prompt | self.llm.with_structured_output(SubQueries)

        input_data = {"query": query, "k": k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries
        print(f"Sub query for comprehensive analysis: {sub_queries}")

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k = 2))

        # use llm to ensure diversity and relevant
        diversity_prompt = PromptTemplate(
            input_variables = ["query", "docs", "k"],
            template = """
                Select the most diverse and relevant set of {k} documents for the query: '{query}'\n Documents: {docs}\n
                Return only indices of select documents as a list of integers.
            """

        )
        diversiry_chain = diversity_prompt | self.llm.with_structured_output(select_indices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]} ..."for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        select_indices_result = diversiry_chain.invoke(input_data).indices
        print(f'selected diverse and relevant documents')

        return [all_docs[i] for i in select_indices_result if i<len(all_docs)]


"""
Define opinion retriever stratergy
"""
class Opinion_retrieval(BaseRetrievalStrategy):
    def retriever(self, query, k = 4):
        # use llm to identify potential viewpoints
        viewpoints_prompt = PromptTemplate(
            input_variables = ["query", "k"],
            template = "Indntify {k} distinct viewpoints or perspectives on the topic: {Query}"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": query, "k": k}
        viewpoints = viewpoints_chain.invoke(input_data)
        print(f"Viewpoints {viewpoints}")

        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k = 2))

        # use llm to classify and diverse opinion
        opinion_prompt = PromptTemplate(
            input_varibables = ["query", "docs", "k"],
            template = "Classify these document into distinct opinion on '{query}' and select the {k} most representative and diverse viewpoints: \n Documents: {docs}\n Selected indices:"
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(select_indices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}


        select_indices = opinion_chain.invoke(input_data).indices
        print("Select diverse and relevant documents")
        return [all_docs[int(i)] for i in select_indices.split() if i.isdigit() and int(i)< len(all_docs)]
"""
Define contextual retrieval stratergy
"""
class Contextual_retrieval(BaseRetrievalStrategy):
    def retrieval(self, query, k = 4, user_context = None):
        # use LLm to incorporate user context into the query
        user_context_prompt = PromptTemplate(
            input_variables= ["query", "context"],
            template = "Given the user context: {Context}\n Reformulate  the query to best address the user's needs: {query}"
        )
        context_chain = user_context_prompt | self.llm
        input_data = {"query": query, "context": user_context or "No specifc context provided"}
        new_context_query = context_chain.invoke(input_data).content
        print(f"contextualized {new_context_query}")

        #  retrieve documents using the contextualized  query
        docs = self.db.similarity_search(new_context_query, k = k*2 )

        # use llm to rank the relevance of retrieved documents considering the user context
        relevant_prompt = PromptTemplate(
            input_varibles = ["query", "context", "doc"],
            template = "Given the query: '{query}' and user context:'{context}', rate the relevance of this document of a scale of 1-10:\n Document: {doc}\nRelevance score:"
        )
        ranking_chain = relevant_score | self.llm.with_structured_output(relevant_score)
        print("")
        ranked_docs = []
        for doc in docs:
            input_data = {"query": new_context_query, "context": user_context or "No specific context provided", "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))


        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:k]]


"""
define  the adaptive retrieval class
"""
class AdaptiveRetriever:
    def __init__(self, texts: List[str]):
        self.classifier = queries_classifier()
        self.strategies = {
            "Factual" : FactualRetrieval(texts),
            "Analytical": AnalyticalRetrieval(texts),
            "Opinion": Opinion_retrieval(texts),
            "Contextual": Contextual_retrieval(texts)
        }
    def get_relevant_documents(self, query: str) -> List[Document]:
        category = self.classifier.classify(query)
        strategy = self.strategies[category]
        return strategy.retrieve(query)

"""
Define additional retriever that inherits from langchain  BaseRetriever       
"""
class PydanticAdaptiveRetriever(BaseRetriever):
    adaptive_retriever : AdaptiveRetriever = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.adaptive_retriever.get_relevant_documents(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


"""
Define Adaptive retriever class
"""

class AdaptiveRAG:
    def __init__(self, texts: List[str]):
        adaptive_retriever = AdaptiveRetriever(texts)
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        
        # Create a custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Create the LLM chain
        self.llm_chain = prompt | self.llm
        
      

    def answer(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        input_data = {"context": "\n".join([doc.page_content for doc in docs]), "question": query}
        return self.llm_chain.invoke(input_data)


if __name__ == "__main__":
        # Usage
    texts = [
        "The Earth is the third planet from the Sun and the only astronomical object known to harbor life."
        ]
    rag_system = AdaptiveRAG(texts)

    factual_result = rag_system.answer("What is the distance between the Earth and the Sun?").content
    print(f"Answer: {factual_result}")

    analytical_result = rag_system.answer("How does the Earth's distance from the Sun affect its climate?").content
    print(f"Answer: {analytical_result}")

    opinion_result = rag_system.answer("What are the different theories about the origin of life on Earth?").content
    print(f"Answer: {opinion_result}")

    contextual_result = rag_system.answer("How does the Earth's position in the Solar System influence its habitability?").content
    print(f"Answer: {contextual_result}")