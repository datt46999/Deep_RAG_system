"""
RAG Evaluation Script
this script discuss  the performance of a RAG system using various metrics from the from the deepeval library.

Dependencies:
- deepeval
- langchain_openai
- json

Custom modules:
- helper_functions (for RAG-specific operations)

"""

from typing import List, Dict, Any

from deepeval import evaluate
from deepeval.test_case import LLMTestCase , LLMTestCaseParams
from deepeval.metrict import GEval, FaithfulnessMetric, ContextualRelevancyMetric


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser #simplest and most common output parser



# from RAG_technique.helper_functions import (
#     create_question_answer_from_context_chain,
#     answer_question_from_context,
#     retrieve_context_per_question
# )

def create_deep_eval_test_cases(
    questions: List[str],
    gt_answers: List[str],
    generated_answers: List[str],
    retriever_documents: List[str]
)->List[LLMTestCase]:
    """
    Create a list of LLMTestCase object for evaluation
    ARG:
    question(List[str]): List of input questions.
    gt_ansert(List[str]): list of group truth answers
    generate_answers(List[str]): List of generate answers
    retriever_documents(Lisr[str]): List of retriever documents

    Return:
    List[LLMTestCase]: List of Test case object
    """

    return [
        LLMTestCase(
            input = question,
            expected_output = gt_answer,
            actual_output = generated_answer,
            retriever_documents = retriever_document
        )
        for question, gt_answer, generated_answer, retriever_document in zip(questions, gt_answers, generated_answers, retriever_documents)
    ]
# define evaluaion metrics


correctness_metric = GEval(
    name="Correctness",
    model="gpt-4-turbo",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4-turbo",
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model="gpt-4-turbo",
    include_reason=True
)

def evaluate_rag(retriever, num_questions: int = 5)-> Dict[str, Any]:
    """
    Evaluation RAG system using predefing test case questions and metrics

    ARG:
    retriever: the Retriever component to evalution
    num_quesiton: number of test question to generate

    Return:
    Dict containing evaluation metrics

    """
    llm = ChatOpenAI(model = "gpt-4-turbo", temperature = 0)
    
    eval_prompt_template = PromptTemplate.from_template("""
        Evaluate the following retrieval results for the question.
        
        Question: {question}
        Retrieved Context: {context}
        
        Rate on a scale of 1-5 (5 being best) for:
        1. Relevance: How relevant is the retrieved information to the question?
        2. Completeness: Does the context contain all necessary information?
        3. Conciseness: Is the retrieved context focused and free of irrelevant information?
        
        Provide ratings in JSON format:
    """)

    # create evaluation chain
    eval_chain = eval_prompt_template | llm | StrOutputParser()

    # eval_chain = (
    #     eval_prompt 
    #     | llm 
    #     | StrOutputParser()
    # )


    # generate test chain
    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser

    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")
    
    #  evaluation each question
    results = []
    for question in questions:
        # Get retrieval results
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Evaluate results
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)
    
    return {
        "questions": questions,
        "results": results,
        "average_scores": calculate_average_scores(results)
    }

def calculate_average_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate average scores across all evaluation results."""
    # Implementation depends on the exact format of your results
    pass

if __name__ == "__main__":
    # Add any necessary setup or configuration here
    # Example: evaluate_rag(your_chunks_query_retriever_function)
    pass