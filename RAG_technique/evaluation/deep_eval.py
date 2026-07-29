
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams



def create_deep_eval_test_cases(questions, gt_answers, generate_answers, retrievered_documents):
    """
    Create Multiple LLMTestCase base on four lists
    """

    return [LLMTestCase(
        input = question,
        expected_output = gt_answer,
        actual_output = generate_answer,
        retrieval_context = retrievered_document
    ) for question, gt_answer, generate_answer, retrievered_document in zip(questions, gt_answers, generate_answers, retrievered_documents)
    ]

"""
test Correctness
"""

# correctness_metric = GEval(
#         name="Correctness",
#         model="gpt-4o",
#         evaluation_params=[
#             LLMTestCaseParams.EXPECTED_OUTPUT,
#             LLMTestCaseParams.ACTUAL_OUTPUT],
#             evaluation_steps=[
#             "Determine whether the actual output is factually correct based on the expected output."
#         ],
#     )


# gt_answer = "Madrid is the capital of Spain."
# pred_answer = "MadriD"

# test_case_correctness = LLMTestCase(
#     input = "What is capital of Spain?",
#     expected_output = gt_answer,
#     actual_output = pred_answer
# )
# correctness_metric.meassure(test_case_correctness)
# print(correctness_metric.score)
  


"""
Test Faithfulness
"""
# question = "What is 3+3?"
# context = ["6"]
# generate_answer = "6"


# faithfulness_metric = FaithfulnessMetric(
#     threshold = 0.7,
#     model = "gpt-4o-mini",
#     include_reason = False
# )

# test_case = LLMTestCase(
#     input = question,
#     actual_output = generate_answer,
#     retriever_context = context
# )

# faithfulness_metric.meassure(test_case)
# print(faithfulness_metric.score)
# print(faithfulness_metric.reason)



"""
Text contextual  relevancy
"""
# actual_output = "then go somewhere else."
# retrieval_context = ["this is a test context","mike is a cat","if the shoes don't fit, then go somewhere else."]
# gt_answer = "if the shoes don't fit, then go somewhere else."


# relevance_metric = ContextualRelevancyMetric(
#     threshold =1 ,
#     model = "gpt-4o-mini",
#     include_reason = True
# )
# relevance_test_case = LLMTestCase(
#     input = "What if these shose don't fit?",
#     actual_output = actual_output,
#     retrieval_context = retrieval_context,
#     expected_output = gt_answer
# )

# relevance_metric.measure(relevance_test_case)
# print(relevance_metric.score)
# print(relevance_metric.reason)
