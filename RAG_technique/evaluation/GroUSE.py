
relevancy_evaluation_prompt = """# Task

Task: Grounded Question Answering
Based solely on the content of the references, the objective is to generate a response to the user's query. Each statement must be followed by the reference of the source passage, in the format [i] where i is the number of the reference. If no passage seems relevant, the answer should begin with "No document seems to precisely answer your question" and may be supplemented with related sourced information.

# Instructions

I will provide you with two answers, numbered 1 and 2, each containing a response to the user request.
I want you to assign to each answer a relevancy grade between 1 and 5:
- Answer relevancy evaluates if the content of the answer accurately responds to the user's question.
- The truthfulness of the information in the answer does not impact relevancy: even if information that appears false is contained in the answer, as long as this information is related to the request, then relevancy should not decrease. Remember that this information could come from references mentioning imaginary content that you are unaware of: the only thing to evaluate to assign the relevancy grade is therefore the adequacy between the information in the answer and the request, NOT their truthfulness.
- The absence of information in the answer does not impact relevancy, only the information contained in the answer is evaluated.
- Answer relevancy cannot be evaluated if the answer mentions that no document responds to the user request, it is then `null`, regardless of whether it contains other information or not.

Rating scale:
null - The answer asserts that no document precisely responds to the user request. Even if it provides additional information, whether appropriate or not, the relevancy remains `null`.
5 - The answer has excellent relevancy. All information provided in the answer is in line with the question and precisely answers the user request.
4 - The answer achieves good relevancy by providing relevant information to answer the user question. Some information indicated does not exactly answer the question, but remains in line with the request.
3 - The answer has average relevancy, it contains information that allows responding to the user request, but it also contains superfluous information, which was not necessary to answer the request.
2 - The answer shows low relevancy, with some elements related to the request, but the majority of the content is not in line with the question asked.
1 - The answer has very low relevancy, not answering the user's question at all. The content is largely inappropriate or off-topic, delivering no useful information for the request.

Before assigning each grade, you will check that the answer does not contain "No document responds...", if this is the case you must put a grade of `null`. If this is not the case, you will then analyze the adequacy between the request and the information contained in the answer.
Your response should be in JSON format, respecting the following format:
{
    "answer_1": {
        "answer_affirms_no_document_answers": X,
        "answer_relevancy_justification": "...",
        "answer_relevancy": Y
    },
    "answer_2": {
        "answer_affirms_no_document_answers": X,
        "answer_relevancy_justification": "...",
        "answer_relevancy": Y
    }
}
Where "..." is a string, X is a boolean, and Y is an integer between 1 and 5 or `null`.

# Sample

User request: {{ input }}

# To evaluate

Answer 1: {{ expected_output }}
Answer 2: {{ actual_output }}"""


"""
=======================================================================
"""
completeness_evaluation_prompt = """# Task

Task: Grounded Question Answering
Based solely on the content of the references, the objective is to generate a response to the user's query. Each statement must be followed by the reference of the source passage, in the format [i] where i is the number of the reference. If no passage seems relevant, the answer should begin with "No document seems to precisely answer your question" and may be supplemented with related sourced information.

# Instructions

I will provide you with two answers, numbered 1 and 2, each containing a response to the user request.
I want you to assign to each answer a completeness grade between 1 and 5:
- The only condition for an answer to be complete is the presence in it of at least all the information from the references that are relevant to the question asked.
- The presence of unrelated information in the answer does not impact completeness.
- The presence of information in the answer not from the references does not impact completeness.
- Possible errors in the sources citing the references do not impact completeness.
- Completeness cannot be evaluated if the references contain no information that can precisely answer the user request, in which case the grade takes the value `null`.

Rating scale:
null - The references contained no relevant information to precisely answer the user's question. In this case, there is no need to read the content of the answer to know that the grade is `null`.
5 - The answer is very complete, it contains all the relevant information from the references. No essential information is omitted, ensuring complete coverage of the question asked.
4 - The answer covers most of the relevant information in depth. It integrates the references satisfactorily, covering the majority of key points. Some details may be missing, but overall, the answer is substantial.
3 - The answer reasonably addresses a number of relevant aspects. It integrates part of the necessary information from the references. However, gaps remain, impacting the overall completeness.
2 - The answer only covers a minimal part of the relevant information. It misses several important information from the references.
1 - The answer covers none of the relevant information, all relevant information from the references has been omitted in the answer.

Before assigning each grade, you will always start by analyzing the information found in the references that are relevant to the user request. If there is no relevant information in the references, completeness must be `null`. If there are relevant information in the references, you will analyze which portion of this information is present or absent in the answers to evaluate the completeness grade. Your response should be in JSON format, respecting the following format:
{
    "answer_1": {
        "completeness_justification": "...",
        "completeness": X
    },
    "answer_2": {
        "completeness_justification": "...",
        "completeness": X
    }
}
Where "..." is a string, and X is an integer between 1 and 5 or `null`.

# SAMPLE

List of references :
{%- for context in contexts %}
Reference {{ loop.index }}: {{ context }}
{%- endfor %}
User request: {{ input }}

# To evaluate

Answer 1: {{ expected_output }}
Answer 2: {{ actual_output }}"""



"""
=======================================================================
"""

faithfulness_evaluation_prompt = """# Task

Task: Grounded Question Answering
Based solely on the content of the references, the objective is to generate a response to the user's query. Each statement must be followed by the reference of the source passage, in the format [i] where i is the number of the reference. If no passage seems relevant, the answer should begin with "No document seems to precisely answer your question" and may be supplemented with related sourced information.

# Instructions

I will provide you with two answers, numbered 1 and 2, each containing a response to the user request.
I want you to assign to each answer a boolean faithfulness grade. An answer is faithful if:
- Each statement made by the answer is followed by a source indicating the reference from which it is drawn.
- The information preceding the source is indeed from the corresponding reference.
- The information preceding the source is in agreement with the corresponding reference, and does not assert facts different from those indicated in the reference.
In all other cases, the response is considered non-faithful.
Faithfulness is also considered non-measurable if the answer asserts that no document responds to the question, and it does not provide any related information, it is then `null`.

Rating scale:
null - The answer asserts that no document responds to the question, and does not provide any related information.
1 - All sentences in the answer cite their sources, and are in agreement with the cited sources.
0 - At least one sentence in the response does not cite its sources, or cites a wrong source, or modifies the content from the references, or asserts something that is not supported by the cited references.

Before assigning each grade, you will start by verifying that the answer does not only assert "No document responds...", without any other information. If this is the case, then faithfulness must be `null`. Otherwise, I want you to analyze by explaining for each sentence, one after the other, if 1) a reference follows the sentence, 2) the reference following the sentence is correct, and 3) if the sentence does not distort or modify the content of the references. Your response should be in JSON format, respecting the following format:
{
    "answer_1": {
        "answer_only_asserts_no_document_answers": X,
        "content_analysis_sentence_by_sentence": [
            {
                "sentence": "...",
                "criterion_1": "...",
                "criterion_2": "...",
                "criterion_3": "..."
            },
            ...
        ],
        "faithfulness_justification": "...",
        "faithfulness": Y
    },
    "answer_2": {
        "answer_only_asserts_no_document_answers": X,
        "content_analysis_sentence_by_sentence": [
            {
            "sentence": "...",
            "criterion_1": "...",
            "criterion_2": "...",
            "criterion_3": "..."
            },
            ...
        ],
        "faithfulness_justification": "...",
        "faithfulness": Y
    }
}
Where "..." is a string, X is a boolean, and Y is either a boolean or `null`.

# Sample

List of references :
{%- for context in contexts %}
Reference {{ loop.index }}: {{ context }}
{%- endfor %}

# To evaluate

Answer 1: {{ expected_output }}
Answer 2: {{ actual_output }}"""


"""
=======================================================================
"""

usefulness_evaluation_prompt = """# Task

Task: Grounded Question Answering
Based solely on the content of the references, the objective is to generate a response to the user's query. Each statement must be followed by the reference of the source passage, in the format [i] where i is the number of the reference. If no passage seems relevant, the answer should begin with "No document seems to precisely answer your question" and may be supplemented with related sourced information.

# Instructions

I will provide you with two answers, numbered 1 and 2, each containing a response to the user request.
I want you to assign to each answer a usefulness grade of 0 or 1:
- Usefulness is only evaluated when the answer says that no document precisely answers the user's question, but it still provides information related to the question.
- Usefulness measures how interesting the related information is to know for the user, given that there is no answer in the references.
- If the answer responds to the user request, usefulness must be `null`.
- If the answer indicates that no document responds to the user request, without adding other information, usefulness must be `null`.

Rating scale:
null - (The answer responds to the user request) OR (the answer does not answer the user's question AND does not provide any related information).
1 - The related information is generally related to the question and adds value to the general understanding of the topic.
0 - The related information is completely off-topic with respect to the question asked.

Before assigning each grade, you will start by verifying that the answer indeed asserts "No document responds...", then you will check that the answer contains related information in addition to this assertion. If one of these two conditions is `false` then usefulness must be `null`.
If both conditions are indeed true, then you will analyze the usefulness of having added this related information to evaluate the usefulness grade. Your response should be in JSON format, respecting the following format:
{
    "answer_1": {
        "answer_affirms_no_document_answers": X,
        "answer_contains_related_information": X,
        "usefulness_justification": "...",
        "usefulness": Y
    },
    "answer_2": {
        "answer_affirms_no_document_answers": X,
        "answer_contains_related_information": X,
        "usefulness_justification": "...",
        "usefulness": Y
    }
}
Where "..." is a string, X is a boolean, and Y is an integer that is 0 or 1 or `null`.

# Sample

User request: {{ input }}

# To evaluate

Answer 1: {{ expected_output }}
Answer 2: {{ actual_output }}"""





import os

import nest_asyncio

from grouse import (
    EvaluationSample,
    GroundedQAEvaluator,
    meta_evaluate_pipeline,
)
from dotenv import load_dotenv
load_dotenv()
prompts_path = "gpt4o_mini_prompts"
os.makedirs(prompts_path, exist_ok=True)

with open(os.path.join(prompts_path, "answer_relevancy.txt.jinja"), "w") as file:
    file.write(relevancy_evaluation_prompt)
with open(os.path.join(prompts_path, "completeness.txt.jinja"), "w") as file:
    file.write(completeness_evaluation_prompt)
with open(os.path.join(prompts_path, "faithfulness.txt.jinja"), "w") as file:
    file.write(faithfulness_evaluation_prompt)
with open(os.path.join(prompts_path, "usefulness.txt.jinja"), "w") as file:
    file.write(usefulness_evaluation_prompt)


meta_evaluations = meta_evaluate_pipeline("gpt-4o-mini", prompts_path, train_set=True)
print("Aggregated metrics")
print(meta_evaluations.report)