"""
This tutorial walks through bulding complete evaluation pipeline for RAG system. Rather than relying on a single matric, we comline multiple evaluation dimentions-completeness,
fectual accuracy and hallucination detection into unified pipeline
"""

"""
Utils
"""
import asyncio
import os
import random
import pandas as pd
from pydantic import BaseModel, Field
from typing import Union, Optional, Type, List
from openai import AsyncOpenAI
from dotenv import load_dotenv


load_dotenv()

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
client = AsyncOpenAI(api_key = OPEN_API_KEY)

async def run_llm_call(system_prompt: str, user_prompt: str, response_model: Optional[Type[BaseModel]], model: str= MODEL_NAME)-> Union[str, None]:
    max_retries = 5
    base_delay = 30

    for attempt in range(1,max_retries+1):
        try:
            if response_model:
                response = await client.beta.chat.completations.parse(
                    model = model,
                    message = [
                        {'role':'system', 'content': system_prompt.strip()},
                        {'role':"user", 'content': user_prompt.strip()}
                    ],
                    temperature = 0,
                    response_format = response_model
                )
                return response.choices[0].message.parsed
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': system_prompt.strip()},
                        {'role': 'user', 'content': user_prompt.strip()},
                    ],
                    temperature=0,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f'Attempt {attempt} - OpenAI Error: {e}')
            if attempt == max_retries:
                return None
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(sleep_time)

async def run_llm_calls(system_prompts: List[str], user_prompts: List[str],
                        response_model: Optional[Type[BaseModel]] = None,
                        model: str = MODEL_NAME) -> List[str]:
    tasks = [
        run_llm_call(system_prompt, user_prompt, model=model,
                     response_model=response_model)
        for system_prompt, user_prompt in zip(system_prompts, user_prompts)
    ]
    return await asyncio.gather(*tasks)



"""
++++++++++++++++++++++++++++++++++++++++++++
CHoosing evaluation criteria
++++++++++++++++++++++++++++++++++++++++++++
"""
df = pd.read_csv('https://figshare.com/ndownloader/files/53919875')
# print(df.head(2))

# using llm to analyze the feedback and summaries the common problem
sys_message_analyze_reasons = """
You are an amazing data analysts analyzing user feedback provided for a questions answering agent response.
Your task to analyze the user feedback and summarize the what are a common problem in the agent respones.
Return up to 5 common problems
""" 
df[df['annotation'] == 'bad']['annotation_reasoning'].sample(5).values.tolist()
only_bad = df[df['annotation']== "bad"].copy()
response = await run_llm_calls(
    system_prompts=[sys_message_analyze_reasons],
    user_prompts=['\n'.join(only_bad['annotation_reasoning'].tolist())],
)
# print(response[0])


"""
++++++++++++++++++++++++++++++++++++++++++++
Build the Benchmark and completeness
++++++++++++++++++++++++++++++++++++++++++++
"""

system_prompt_complete = """
You are evaluating  wheather a response from a question answering agent is incomplete.
Your task: Base on the user feedback, determine if the primary issue with the agent's answer is a **completeness problem**-i.e., 
It is a missing key imformation that should have been included. If the issue is due to something else (e.g., hallucination, contradition, pool phrasing), mask it as no relate to completeness.
Respond with clear Yes/No judgment and a brief reasoning.
""".strip()
user_prompt_reasoning_eval = """
Question:
{question}

Wrong Answer:
{output}

Annotation Reasoning:
{reasoning}
""".strip()



class CompletenessRelated(BaseModel):
    reasoning: str
    has_completeness_problem: bool

sys_msgs = [system_prompt_complete]*df.shape[0]
user_msgs = [
    user_prompt_reasoning_eval.format(
        question=row['input'],
        output=row['output'],
        reasoning=row['annotation_reasoning'],
    )
    for _, row in only_bad.iterrows()
]
responses = await run_llm_calls(sys_msgs, user_msgs, response_model=CompletenessRelated)
only_bad['completeness_related'] = [r.has_completeness_problem for r in responses]
only_good = df[df['annotation'] == 'good'].copy()
completeness_eval_df = pd.concat([only_good.sample(20, random_state=1), only_bad[only_bad['completeness_related']]])
completeness_eval_df['annotation'] = completeness_eval_df['annotation'].replace({'good': 1, 'bad': 0})

# print(completeness_eval_df.shape)
# print(completeness_eval_df.head(2))


"""
++++++++++++++++++++++++++++++++++++++++++++

Completeness Matric

++++++++++++++++++++++++++++++++++++++++++++
"""
from enum import Enum
from sklearn.metrics import classification_report

# system_prompt_completeness_eval = """
# Decide if the answer is missing important informatio base on the question.
# Is the answer complete? Answer complete or incomplete.
# """.strip()

# user_prompt_answer_eval = """
# Question:
# {question}

# Context:
# {context}

# Answer:
# {conclusions}
# """.strip()
# class ScoreValue(int, Enum):
#     INCOMPLETE = 0
#     COMPLETE = 1

# class CompletenessScore(BaseModel):
#     score: ScoreValue
# sys_msgs = [system_prompt_completeness_eval] * completeness_eval_df.shape[0]
# user_msgs = [
#     user_prompt_answer_eval.format(
#         question=row['input'],
#         context=row['information_retrieved'],
#         conclusions=row['output']
#     )
#     for _, row in completeness_eval_df.iterrows()
# ]
# responses = await run_llm_calls(sys_msgs, user_msgs, response_model=CompletenessScore)
# completeness_eval_df['naive_method'] = [int(x.score.value) for x in responses]

# # print(completeness_eval_df.naive_method.value_counts())
# classification_report(
#     completeness_eval_df.annotation,
#     completeness_eval_df.naive_method,
#     output_dict=True,
#     zero_division=0
# )


system_prompt_completeness_eval = """
You are an assistant evaluating how complete an answer is, given a question and supporting context.

First, consider what are the key pieces of information a fully complete answer should include based on the question and context.  
Then, check whether the answer contains all of that information.

Provide a short explanation of your reasoning.  
Then assign a score:

0 - Not complete: Key information is missing or major parts of the question are not addressed.  
1 - Partially complete: All parts of the question are touched on, but some are incomplete or only loosely supported by the context.  
2 - Fully complete: The answer thoroughly and directly addresses all aspects of the question, using information clearly supported by the context.
""".strip()

class ScoreValue(int, Enum):
    INCOMPLETE = 0
    PARTIALLY_COMPLETE = 1
    COMPLETE = 2

class CompletenessScore(BaseModel):
    reasoning: str
    score: ScoreValue


sys_msgs = [system_prompt_completeness_eval] * completeness_eval_df.shape[0]
responses = await run_llm_calls(sys_msgs, user_msgs, response_model=CompletenessScore, model="gpt-4.1-mini")
completeness_eval_df['advenced_method'] = [int(x.score > 1) for x in responses]
completeness_eval_df['advenced_method_reasoning'] = [x.reasoning for x in responses]

print(completeness_eval_df.advenced_method.value_counts())
classification_report(
    completeness_eval_df.annotation,
    completeness_eval_df.advenced_method,
    output_dict=True,
    zero_division=0
)


"""
++++++++++++++++++++++++++++++++++++++++++++

Utilize RAGAS for hallucination detection

++++++++++++++++++++++++++++++++++++++++++++
"""

system_prompt_is_factually_correct = """
You are evaluating whether a response from a question-answering agent has a **factual correctness problem**.

Your task: Based on the user feedback, determine if the main issue with the answer is related to **hallucination** (made-up or incorrect information) or **contradiction** (statements that conflict with known facts or the context). If the issue is something else (e.g., incomplete, vague, poorly phrased), mark it as not related to factual correctness.

Respond with a clear yes/no judgment and a brief explanation.
""".strip()
class FactualRelated(BaseModel):
    reasoning: str
    has_factual_correctness_problem: bool


sys_msgs = [system_prompt_is_factually_correct] * df.shape[0]
user_msgs = [
    user_prompt_reasoning_eval.format(
        question=row['input'],
        output=row['output'],
        reasoning=row['annotation_reasoning'],
    )
    for _, row in only_bad.iterrows()
]
responses = await run_llm_calls(sys_msgs, user_msgs, response_model=FactualRelated)
only_bad['factual_related'] = [r.has_factual_correctness_problem for r in responses]
print(only_bad['factual_related'].value_counts())

factual_eval_df = pd.concat([only_good.sample(sum(only_bad['factual_related']), random_state=1), 
                             only_bad[only_bad['factual_related']]])
factual_eval_df['annotation'] = factual_eval_df['annotation'].replace({'good': 1, 'bad': 0})

# Test RAGAS

# import os
# from ragas.metrics import faithfulness
# from ragas.dataset_schema import SingleTurnSample
# from langchain.chat_models import ChatOpenAI
# from ragas import evaluate, EvaluationDataset

# langchain_client = ChatOpenAI(
#     model_name="gpt-4.1",
#     openai_api_key=os.environ["OPENAI_API_KEY"],
#     temperature=0
# )

# samples = [SingleTurnSample(
#     user_input=row['input'],
#     response=row['output'],
#     retrieved_contexts=[row['information_retrieved']],
# ) for _, row in factual_eval_df.iterrows()]

# results = evaluate(EvaluationDataset(samples=samples), metrics=[faithfulness], llm=langchain_client)
# factual_eval_df['ragas_faithfulness'] = [r['faithfulness'] for r in results.scores]
# factual_eval_df['ragas_faithfulness_binary'] = [1 if r['faithfulness'] > 0.5 else 0 for r in results.scores]
# print(factual_eval_df.ragas_faithfulness_binary.value_counts())

# classification_report(
#     factual_eval_df.annotation,
#     factual_eval_df.ragas_faithfulness_binary,
#     output_dict=True,
# )


"""
++++++++++++++++++++++++++++++++++++++++++++

Aggregate to end2end evaluationg pipeline

++++++++++++++++++++++++++++++++++++++++++++
"""
#  Is this example we will only take into account evaluation of completeness and factual correctness
# We will use the advanced method for completeness and the RAGAS for factual correctness

def get_final_annotation(row):
    factual_score = evalaute_fact   uality_via_ragas(row)
    completeness_score = evalaute_completeness_via_llm_as_judge(row)

    if factual_score < FACTUALITY_THRESHOLD:
        return "bad", "factuality problem"
    elif completeness_score < COMPLETENESS_THRESHOLD:
        return "bad", "completeness problem"
    return "good", ""


df['automated_annotation_pipeline'] = df.apply(get_final_annotation, axis=1)
classification_report(
    df.annotation,
    df.automated_annotation_pipeline,
    output_dict=True,
)

# orther evalatiuon
# https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/evaluation/open-rag-eval-example.ipynb