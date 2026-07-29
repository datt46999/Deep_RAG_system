import os
from  dotenv import load_dotenv

from grouse import (
    EvaluationSample,
    GroundedQAEvaluator,
    meta_evaluate_pipeline,
)

load_dotenv()
evaluator = GroundedQAEvaluator()


# 


"""
Evaluate a good answer
"""
# good_sample = EvaluationSample(
#     input="Where is the Eiffel Tower located?",
#     actual_output="The Eiffel Tower stands in the Champs de Mars in Paris.[1]",
#     expected_output="In the Champs de Mars in Paris. [1]",
#     references=[
#         "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France"
#     ]
# )

# result = evaluator.evaluate(eval_samples = [good_sample]).evaluations[0]
# print("Answer Relevancy (1 to 5):", result.answer_relevancy.answer_relevancy)
# print("Answer Relevancy (1 to 5):", result.answer_relevancy.answer_relevancy_justification)
# print("Completeness (1 to 5):", result.completeness.completeness)
# print("Completeness (1 to 5):", result.completeness.completeness_justification)
# print("Faithfulness (0 or 1):", result.faithfulness.faithfulness)
# print("Faithfulness (0 or 1):", result.faithfulness.faithfulness_justification)


# Answer Relevancy (1 to 5): 5
# Answer Relevancy (1 to 5): The answer directly responds to the user's question about the location of the Eiffel Tower.
# Completeness (1 to 5): 5
# Completeness (1 to 5): The answer includes all the relevant information from the reference, which is the location of the Eiffel Tower.
# Faithfulness (0 or 1): 1
# Faithfulness (0 or 1): The sentence in the answer cites the correct source and does not distort or modify the content of the reference.
# 100%|█████████████████████████████████████████████| 1/1 [00:34<00:00, 34.87s/it]
# 2026-07-29 21:31:32,103 - LLM Call Tracker - INFO - Cost: 0.2419$
# Usefulness (0 or 1): 1
# Justification: The related information provided is directly related to the question, as it mentions a petition against the Eiffel Tower construction, which implies criticism of the project.
# Positive Acceptance: 1
# Negative Rejection: 1


"""
how does it behave with an irrelevant answer
"""

irrelevant_sample = EvaluationSample(
    input = "Where is the Eiffel Tower located?",
    actual_output = "The Eiffel Tower is mainly made of puddle iron.[2]",
    expected_output = "In the Champs de Mars in Paris.[1]",
    references =[
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France",
        "The puddle iron (wrought iron) of the Eiffel Tower weighs 7,300 tonnes,[70] and the addition of lifts, shops and antennae have brought the total weight to approximately 10,100 tonnes."
    ]
)
result = evaluator.evaluate(eval_samples = [irrelevant_sample]).evaluations[0]

print("Answer Relevancy (1 to 5):", result.answer_relevancy.answer_relevancy)
print("Justification:", result.answer_relevancy.answer_relevancy_justification)



"""
Evaluation of an incomplete sample
# """
# incomplete_sample = EvaluationSample(
#     input="Who critized the Eiffel Tower project in 1889?",
#     actual_output=(
#         "The tower was critized by those who did not believe it was feasible and some artists.[1]"
#     ),
#     expected_output=(
#         "The tower was critized by those who did not believe it was feasible and those who objected on artistic grounds.[1]"
#         "An artist committee was created to protest againt the construction of the tower, led by the prominent architect "
#         "Charles Garnier and including some of the most important figures of the arts, "
#         "such as William-Adolphe Bouguereau, Guy de Maupassant, Charles Gounod and Jules Massenet. [2]"
#     ),
#     references=[
#         "The proposed tower had been a subject of controversy, drawing criticism from those who did not believe it was feasible and those who objected on artistic grounds.",
#         (
#             "It came to a head as work began at the Champ de Mars: a \"Committee of Three Hundred\" "
#             "(one member for each metre of the tower's height) was formed, led by the prominent architect "
#             "Charles Garnier and including some of the most important figures of the arts, "
#             "such as William-Adolphe Bouguereau, Guy de Maupassant, Charles Gounod and Jules Massenet."
#         ),
#         "A petition called \"Artists against the Eiffel Tower\" was sent to the Minister of Works and Commissioner for the Exposition, Adolphe Alphand, and it was published by Le Temps on 14 February 1887"
#     ]
# )

# result = evaluator.evaluate(eval_samples=[incomplete_sample]).evaluations[0]

# print("Completeness (1 to 5):", result.completeness.completeness)
# print("Justification:", result.completeness.completeness_justification)


# Answer Relevancy (1 to 5): 1
# Justification: The answer provides information about the material of the Eiffel Tower, which is not relevant to the user's question about its location.

"""
Evaluation of information utility in case there is no answer to the question in the reference
"""

useful_sample = EvaluationSample(
    input="Who critized the Eiffel Tower project in 1889?",
    actual_output=(
        "No document seems to precisely answer your question."
        "However, it is mentioned that a petition against tht Eiffel Tower construciton was sent "
        "to the Minister of Works and Commissioner for the Exposition [1]"
    ),
    expected_output=(
        "No document seems to precisely answer your question."
        "However, it is worth noting that a petition against tht Eiffel Tower construciton was sent "
        "to the Minister of Works and Commissioner for the Exposition [1]"
    ),
    references=[
        "A petition against the tower was sent to the Minister of Works and Commissioner for the Exposition, Adolphe Alphand, and it was published by Le Temps on 14 February 1887"
    ]
)

result = evaluator.evaluate(eval_samples=[useful_sample]).evaluations[0]

print("Usefulness (0 or 1):", result.usefulness.usefulness)
print("Justification:", result.usefulness.usefulness_justification)
print("Positive Acceptance:", result.positive_acceptance)
print("Negative Rejection:", result.negative_rejection)


# Answer Relevancy (1 to 5): 1
# Justification: The answer provides information about the material of the Eiffel Tower, which is not relevant to the user's question about its location.
# 100%|████████████████████████████████████████████| 1/1 [00:00<00:00, 138.21it/s]
# 2026-07-29 21:46:00,672 - LLM Call Tracker - INFO - Cost: 0.2466$
# Usefulness (0 or 1): 1
# Justification: The related information provided is directly related to the question, as it mentions a petition against the Eiffel Tower construction, which implies criticism of the project.
# Positive Acceptance: 1
# Negative Rejection: 1
"""
Note that all the results are cached and we can compute the global statistics on all the samples
"""
# evaluation_report = evaluator.evaluate(eval_samples=[
#     good_sample,
#     irrelevant_sample,
#     incomplete_sample,
#     unfaithful_sample,
#     useful_sample,
# ]).report
# print("Average answer relevancy: ", evaluation_report.answer_relevancy)
# print("Average completeness: ", evaluation_report.completeness)
# print("Average faithfulness: ", evaluation_report.faithfulness)
# print("Average usefulness: ", evaluation_report.usefulness)