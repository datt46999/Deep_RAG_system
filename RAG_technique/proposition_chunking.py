"""
Implement propotions chunking methed to build rag system

method detail
1: load enviroment variables
2: document chunking
3: proposition generation
4: Quality check
5: Embedding proposition
6: retrieval and comparation


Implement:

Granularity: breaking down document into small factual propositions,
 the system allows for highly specific retrieval, making it easier to extract precise answers from large or complex documents.

Quality Checking: The generation proposition are  passed through a grading system that evaluates accuracy, clarity, completeness and conciseness

VectorStore retrieval: using FAISS after embeddin
Query Testing: Multiple test queries are made to the vector stores (proposition-based and larger chunks) to compare the retrieval performance.

"""

import os


from dotenv import load_dotenv
from typing import List


from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embedding import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq


load_dotenv()


sample_content = """Paul Graham's essay "Founder Mode," published in September 2024, challenges conventional wisdom about scaling startups, arguing that founders should maintain their unique management style rather than adopting traditional corporate practices as their companies grow.
Conventional Wisdom vs. Founder Mode
The essay argues that the traditional advice given to growing companies—hiring good people and giving them autonomy—often fails when applied to startups.
This approach, suitable for established companies, can be detrimental to startups where the founder's vision and direct involvement are crucial. "Founder Mode" is presented as an emerging paradigm that is not yet fully understood or documented, contrasting with the conventional "manager mode" often advised by business schools and professional managers.
Unique Founder Abilities
Founders possess unique insights and abilities that professional managers do not, primarily because they have a deep understanding of their company's vision and culture.
Graham suggests that founders should leverage these strengths rather than conform to traditional managerial practices. "Founder Mode" is an emerging paradigm that is not yet fully understood or documented, with Graham hoping that over time, it will become as well-understood as the traditional manager mode, allowing founders to maintain their unique approach even as their companies scale.
Challenges of Scaling Startups
As startups grow, there is a common belief that they must transition to a more structured managerial approach. However, many founders have found this transition problematic, as it often leads to a loss of the innovative and agile spirit that drove the startup's initial success.
Brian Chesky, co-founder of Airbnb, shared his experience of being advised to run the company in a traditional managerial style, which led to poor outcomes. He eventually found success by adopting a different approach, influenced by how Steve Jobs managed Apple.
Steve Jobs' Management Style
Steve Jobs' management approach at Apple served as inspiration for Brian Chesky's "Founder Mode" at Airbnb. One notable practice was Jobs' annual retreat for the 100 most important people at Apple, regardless of their position on the organizational chart
. This unconventional method allowed Jobs to maintain a startup-like environment even as Apple grew, fostering innovation and direct communication across hierarchical levels. Such practices emphasize the importance of founders staying deeply involved in their companies' operations, challenging the traditional notion of delegating responsibilities to professional managers as companies scale.
"""

embedding_model = OllamaEmbeddings(model='nomic-embed-text:v1.5', show_progress=True)
docs_list = [Document(page_content=sample_content, metadata={"Title": "Paul Graham's Founder Mode Essay", "Source": "https://www.perplexity.ai/page/paul-graham-s-founder-mode-ess-t9TCyvkqRiyMQJWsHr0fnQ"})]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=50
    )
doc_splits = text_splitter.split_documents(docs_list)

for i, doc in enumerate(doc_splits):
    doc.metadata['chunk_id'] = i+1


class GeneratePropositions(BaseModel):
    """
    List of all the propositions in given docment
    """
    propositions : List[str] = Field(
        description="List of propositions (factual, self-contained, and concise information)"
    )


class GradePropositions(BaseModel):
    """Grade a given proposion on accuracy, clarity, completeness, and conciseness"""
    accracy :  int =Field(
        description = "Rate from 1-10 base on how well the proposition reflect the original text"
    ) 
    clarity : int = Field(
        description = "Rate from 1-10 base on how easy to understand base proposition includes necessary details (e.g, date, qualifiers)"
    )
    completeness: int = Field(
        description="Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers)."
    )

    conciseness: int = Field(
        description="Rate from 1-10 based on whether the proposition is concise without losing important information."
    )


def proposition():
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    structure_llm = llm.with_structured_output(GeneratePropositions)

    propotion_examples =[
        {"document": 
            "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.", 
        "propositions": 
            "['Neil Armstrong was an astronaut.', 'Neil Armstrong walked on the Moon in 1969.', 'Neil Armstrong was the first person to walk on the Moon.', 'Neil Armstrong walked on the Moon during the Apollo 11 mission.', 'The Apollo 11 mission occurred in 1969.']"
        },
    ]

    example_proposion_prompt =[
        {
            ("human","{document}"),
            ("ai","{proposition}")
        }
    ]
    few_shot_prompt = FewShotChatMessagePromptTemplate.from_messages(
        example_prompt = example_proposion_prompt,
        examples = propotion_examples
    )


    prompt_system = """
        Please break down the following text into sample, self-contained proposition. Ensure that each proposition meets more the following criteria:

        1. Express a Single Fact: Each proposition should be one specific fact or claim.
        2. Be Understandable without context: The proposition should be self-contained, meaning it can be understood without needing additional context.
        3. Use Full Names, Not Pronouns: Advoid pronouns or ambirous reference, use full entity name.
        4. Include relevant dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
        5. Contain one subject-predicate Relationship: Focus on a single subject and its corresponding action and attribute, without confuntion and multiple clause
    """ 
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', prompt_system),
            few_shot_prompt,
            ('human', '{document}')
        ]
    )
    return prompt | structure_llm

def grade_propotion():
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    structured_llm= llm.with_structured_output(GradePropositions)

    evaluation_prompt_template = """
    Please evaluate the following proposition based on the criteria below:
    - **Accuracy**: Rate from 1-10 based on how well the proposition reflects the original text.
    - **Clarity**: Rate from 1-10 based on how easy it is to understand the proposition without additional context.
    - **Completeness**: Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers).
    - **Conciseness**: Rate from 1-10 based on whether the proposition is concise without losing important information.

    Example:
    Docs: In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.

    Propositons_1: Neil Armstrong was an astronaut.
    Evaluation_1: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

    Propositons_2: Neil Armstrong walked on the Moon in 1969.
    Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

    Propositons_3: Neil Armstrong was the first person to walk on the Moon.
    Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

    Propositons_4: Neil Armstrong walked on the Moon during the Apollo 11 mission.
    Evaluation_4: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

    Propositons_5: The Apollo 11 mission occurred in 1969.
    Evaluation_5: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

    Format:
    Proposition: "{proposition}"
    Original Text: "{original_text}"
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", evaluation_prompt_template),
            ("human", "{proposition}, {original_text}"),
        ]
    )
    return prompt | structured_llm


def evaluate_proposition(proposition, original_text):
    p_evaluation = grade_propotion()
    response = p_evaluation.invoke({"proposition": proposition, "original_text": original_text})
    scores = {"accuracy": response.accuracy, "clarity": response.clarity, "completeness": response.completeness, "conciseness": response.conciseness}  # Implement function to extract scores from the LLM response
    return scores

def passes_quality_check(scores, thresholds):
    for category, score in scores.items():
        if score < thresholds[category]:
            return False
    return True

def define_evaluation():
    evaluation_categories = ["accuracy", "clarity", "completeness", "conciseness"]
    thresholds = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}


    evaluated_propositions = []
    for idx, proposition in enumerate(propositions):
        scores = evaluate_proposition(proposition.page_content, doc_splits[proposition.metadata['chunk_id'] - 1].page_content)
    if passes_quality_check(scores, thresholds):
        # Proposition passes quality check, keep it
        evaluated_propositions.append(proposition)
    else:
        # Proposition fails, discard or flag for further review
        print(f"{idx+1}) Propostion: {proposition.page_content} \n Scores: {scores}")
        print("Fail")
    return evaluated_propositions
def embedding2vector(doc_inputs):
    # doc_inputs can :evaluate_proposition or doc_split 
    vectorstore= FAISS.from_documents(doc_inputs, embedding_model)
    retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}, # number of documents to retrieve
            )
    return retriever

if __name__ == "__main__":
    proposition_generation = proposition()
    propositions = []
    for i in range(len(doc_splits)):
        response = proposition_generation.invoke({"document": doc_splits[i].page_content})
        for p in response.proposition:
            propositions.append(Document(page_content = p, metadata = {"Title": "Paul Graham's Founder Mode Essay", "Source": "https://www.perplexity.ai/page/paul-graham-s-founder-mode-ess-t9TCyvkqRiyMQJWsHr0fnQ", "chunk_id": i+1}))
    

    # testing:
    evaluated_propositions = define_evaluation()

    test_query_1 = "what is the essay \"Founder Mode\" about?"
    test_query_2 = "who is the co-founder of Airbnb?"
    test_query_3 = "when was the essay \"founder mode\" published?"
    retriever_larger= embedding2vector(doc_splits)
    retriever_propositions = embedding2vector(evaluated_propositions)
    res_proposition = retriever_propositions.invoke(test_query_3)
    res_larger = retriever_larger.invoke(test_query_3)
    print("---"*50)
    for i, r in enumerate(res_proposition):
        print(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

    print("---"*50)
    print("\n")
    print("---"*50)
    for i, r in enumerate(res_larger):
        print(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

