"""
Implemants an Explainable Retriever, system that not only retrieves relevant document based on query but also provides explanation for why each retrieved document is relevant.
It combines vector-based similarity with natural language explanations, enchacing the transparentcy and interpretability of the retrieval    process.


Key components
1. Vector store create from input text.
2. Base retriever using FAISS for efficient similarity search.
3. LLM for generating explanations
4. custom explainableRetrieval class that conbines retrieval and explanation generation




"""

import os
from dotenv import load_dotenv

from helper_functions import *
from evaluation.evalute_rag import *

from langchain_openai import ChatOpenAI, OpenAIEmbeddinds
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import FAISS





class ExplainableRetriever:
    def __init__(self, text):
        self.embedding = OpenAIEmbeddinds()
        self.vectorstore = FAISS.from_text(text, self.embedding)
        self.llm = ChatOpenAI(model = "gpt4o-mini", temperature = 0, max_token = 4000)

        # create base retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs = {"k": 5})

        # create an explanation chain
        explain_prompt = PromptTemplate(
            input_variables = ['query', "context"],
            template ="""
                Analyze the relationship between the following query and the retrieved context.
                Explain why this context is relevant to query and how it might help answer the query.
                Query: {query},
                Context: {context},
                Explanation:
            """
        )        
        self.explain_chain = explain_prompt | self.llm 

    def retrieve_explainable(self, query):
        docs = self.retriever.invoke(query)

        explain_results = []
        for doc in docs:
            input_data =  {"query": query, "context": doc.page_context}
            explaination = self.explain_chain.invoke(input_data).content
            explain_results.append({
                "content":doc.page_content,
                "explanation": explaination
            })

        return explain_results


if __name__ == "__main__":
    texts = [
        "The sky is blue because of the way sunlight interacts with the atmosphere.",
        "Photosynthesis is the process by which plants use sunlight to produce energy.",
        "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
    ]

    explainable_retriever = ExplainableRetriever(texts)
    query = "Why is the sky blue?"
    results = explainable_retriever.retrieve_and_explain(query)

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {result['content']}")
        print(f"Explanation: {result['explanation']}")
        print()