import json
import os
from tqdm import tqdm
from typing import Optional
import datasets
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from ragatouille import RAGPretrainedModel
from langchain_core.vectorstores import VectorStore
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

from rag_system import answer_with_rag
from embedding import RAW_KNOWLEDGE_BASE, load_embeddings


from langchain_community.llms import HuggingFaceHub



eval_dataset = datasets.Dataset.from_pandas(
    generated_questions, split="train", preserve_index=False
)
EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

OPENAI_API_KEY = ""

eval_chat_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0, openai_api_key=OPENAI_API_KEY)
evaluator_name = "GPT4"



repo_id = "HuggingFaceH4/zephyr-7b-beta"
READER_MODEL_NAME = "zephyr-7b-beta"
HF_API_TOKEN = ""

READER_LLM = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    huggingfacehub_api_token=HF_API_TOKEN,
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm,
    knowledge_index: VectorStore,
    output_file: str,
    reranker: Optional[RAGPretrainedModel] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_with_rag(
            question, llm, knowledge_index, reranker=reranker
        )
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)


def evaluate_answers(
        answer_path: str,
        eval_chat_model,
        evaluator_name: str,
        evaluation_prompt_template: ChatPromptTemplate,
    )->None:
    """
    Evaluation generated answers. Modifies the given answer file in place for better checkpointing
    """
    answer = []
    # load previous generations if they exists
    if os.path.isfile(answer_path):
        answers = json.load(open(answer_path, "r"))


    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)



if __name__ =="__main__":
    if not os.path.exists("./output"):
        os.mkdir("./output")

    for chunk_size in [200]:  # Add other chunk sizes (in tokens) as needed
        for embeddings in ["thenlper/gte-small"]:  # Add other embeddings as needed
            for rerank in [True, False]:
                settings_name = f"chunk:{chunk_size}_embeddings:{embeddings.replace('/', '~')}_rerank:{rerank}_reader-model:{READER_MODEL_NAME}"
                output_file_name = f"./output/rag_{settings_name}.json"

                print(f"Running evaluation for {settings_name}:")

                print("Loading knowledge base embeddings...")
                knowledge_index = load_embeddings(
                    RAW_KNOWLEDGE_BASE,
                    chunk_size=chunk_size,
                    embedding_model_name=embeddings,
                )

                print("Running RAG...")
                reranker = (
                    RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
                    if rerank
                    else None
                )
                run_rag_tests(
                    eval_dataset=eval_dataset,
                    llm=READER_LLM,
                    knowledge_index=knowledge_index,
                    output_file=output_file_name,
                    reranker=reranker,
                    verbose=False,
                    test_settings=settings_name,
                )

                print("Running evaluation...")
                evaluate_answers(
                    output_file_name,
                    eval_chat_model,
                    evaluator_name,
                    evaluation_prompt_template,
                )