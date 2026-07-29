"""
 Implment RAG approach  with an interagted feed backloop. 
 It aim to improve the quality of relevance over time by incorporate user feed back and dynamically adjusting the retrieval process

 Key components:
1. PDF content extraction: extract text from PDF documents
2. Vectorstore: store and indexes document embedding for efficient retrieval
3. Retriever: fetches relevant documents base on user query
4. languare model: generate response using retriever documents
5. Feetback Collection: Gathers user feedback on response quality and relevance
6. Feetback Storate: presists user feedback future user
7. relevance score adjustment: Modifile document relevance base on feeb back
8. index Fine tune: Periodically updates the vectorstore using accumulated feedback

"""         


import os
import sys
import json
from typing import List, Any, Dict
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from helper_functions import *

load_dotenv()

# function to format user feedback
def get_user_feedback(query, response, relevance, quality, comments =""):
    return {
        "query": query,
        "response": response, 
        "relevance": int(relevance),
        "quality": int(quality),
        "comment": comments
    }

# function to store feedback
def store_feedback(feedback):
    with open("data/feedback_data1.json" , 'a') as f:
        json.dump(feedback, f)
        f.write("\n")

# function to read feedback file
def load_feedback_data():
    feedback_data = []
    try:
        with open("data/feedback_data1.json", 'r') as f:
            for line in f:
                line  = line.strip()
                if not line:
                    continue
                feedback_data.append(json.loads(line))
    except FileNotFoundError:
        print("feedback file found. Starting with empty feedback")
    return feedback_data



# function to adjust  files relevancy base on the feedback file

class Response(BaseModel):
    answer: str = Field(..., title = "The answer to the question. The option can be only 'Yes' or 'No'")


def adjust_relevance_scores(query:str, docs: List[Any], feedback_data: List[Dict[str, Any]]):
    # create a prompt template for relevance checking
    relevance_prompt = PromptTemplate(
        input_variables = ["query", "feedback_query", "doc_query", "feedback_response"],
        template = """
            Determine if the following feedback response is relavant to the current query and document content.
            You are also provide with the query original query that was use to generate the feedback response.
            Current query:{query}
            Feedback query: {feedback_query}
            Document content: {doc_content}
            Feedback response: {feedback_response}

            Is this feedback relevant? Respond with only 'Yes' or 'No'.
        """
    )

    
    llm = ChatOpenAI(model ="gpt-4o-mini", temperature = 0)

    relevant_chain = relevance_prompt | llm.with_structured_output(Response)

    for doc in docs:
        relevant_feedback = []
        for feedback in feedback_data:
            # use LLM to check relevance
            input_chain  ={
                "query" : query,
                "feedback_query" : feedback["query"],
                "doc_content":doc.page_content[:1000],
                "feedback_response": feedback["response"]
            } 
            results = relevant_chain.invoke(input_chain).answer

            if results == "Yes":
                relevant_feedback.append(feedback)
        if relevant_feedback:
            avg_relevance = sum(f["relevance"] for f in relevant_feedback) / len(relevant_feedback)
            doc.metadata["relevance_score"] *= (avg_relevance/3) # Assuming a 1-5 scale, 3 is neural

    # re-rank document base on adjust score
    return sorted(docs, key = lambda x: x.metadata["relevance_score"], reverse = True)


# Function to fine tune the vector store to include also queries and answers that received good feedback
def fine_tune_index(feedback_data: List[Dict[str, Any]], texts: List[str]) -> Any:
    filter_feedback = [f for f in feedback_data if [f["relevance"]>= 4 and f["quality"]>=4]]

    additional_text =[]
    for feedback in filter_feedback:
        combine_feedback = feedback["query"] + feedback["response"]
        additional_text.append(combine_feedback)

    additional_text = " ".join(additional_text)

    all_text = texts + additional_text
    vectorstore = encode_from_string(all_text)
    return vectorstore


prompt_feedback = """
    You are an assistant for question-answering task. Use only the following retrieved.
    If the context does not contain the answer, say "I' don't have enough information to return your answe of you question." and don't make up fact outside the context.
    Context:
    {context}

    Question: {question}

    Answer:
"""


if __name__ =="__main__":
    path = "data/Understanding_Climate_Change.pdf"
    query = "What is greenhourse effect?"

    
    content = read_pdf_of_string (path)
    vectorstore = encode_from_string(content)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    response = qa_chain(query)["result"]
    relevance = 5
    quality = 5
    feedback = get_user_feedback(query, response, relevance, quality)

    store_feedback(feedback)

    docs = retriever.invoke(query)
    adjusted_docs =adjust_relevance_scores(query, docs, load_feedback_data())

    # Update the retriever with adjusted docs
    retriever.search_kwargs['k'] = len(adjusted_docs)
    retriever.search_kwargs['docs'] = adjusted_docs

    # Periodically (e.g., daily or weekly), fine-tune the index
    new_vectorstore = fine_tune_index(load_feedback_data(), content)
    retriever = new_vectorstore.as_retriever()


    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        # chain_type_kwargs={"prompt": prompt_feedback},
        return_source_documents=True,  # hữu ích để debug relevance sau này
    )
    result = qa_chain.invoke(query)
    response = result["result"]
    print(response)
