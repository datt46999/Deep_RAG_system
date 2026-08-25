"""
MemoRAG is a RAG Framework the incorproates a memory model as a rexiliary step before the retriaval plase. 
In doing so, it bridges the gap in contextual understanding and reasoning that standard RAG techniques face when addressing queries with implicits or ambiguous imformation needs and unstructure external knowledge.


Method Detail
1. Memory:
    + The memory module serves as an auxiliary component to enhance the retriever’s ability to identify better matches between queries and relevant parts of the database. 
    It takes the original query and the database as inputs and produces staging answers — intermediate outputs like clues, surrogate queries, or key points — which the retriever uses instead of the original query.
    + Long-term memory is constructed by running a long-context model, such as Qwen2-7B-Instruct or Mistral-7B-Instruct-v0.2, over the entire database. 
    This process generates a compressed representation of the database through an attention mechanism.
    + The compressed representation is stored as key-value pairs (kv-cache), facilitating efficient and accurate retrieval.
    +Released memory models include memorag-qwen2-7b-inst and memorag-mistral-7b-inst, derived from Qwen2-7B-Instruct and Mistral-7B-Instruct-v0.2, respectively.

2. Retriever:
   + The retriever is a standard retrieval model, adapted to take processed queries (created by the memory module as staging answers) instead of the original query.
   + It outputs the retrieved context, which serves as the basis for generating the final answer.
3. Generator:
    + The generate produced the final response by combining the retriever's output(retrieved context) with the original query
    + MemoRAG ensures compatibility and consistency by using the memory module’s underlying model as the default generator.
"""

import os
import pandas as pd
import json
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI
from helper_functions import *


load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class KeyValuePair(BaseModel):
    """
    The key-value pairs are the 'holder' memory.
    The keys are topics from the relevant article and the values are details about that topic
    """
    topic: str
    details: str

class MemoryResponse(BaseModel):
    """The obtain memory is a list topic-details pairs"""
    pairs: List[KeyValuePair]

memory_store_system_prompt = """
    You are an expect at extracting structure key-value pairs from text.
    Identify key the topic, entities, or question and pair them with relevant, detailed information.
    Ensure output is well-structure, advoid redundancy while preserving meaning
"""
memory_store_kv_cache_prompt = """
    You are provided with long article, chunk by chunk. Read each chunk and extract carefully key topics and their detail information.
    For each key topic:
        + Identify the main context, entity, or potential question.
        + Provide the corresponding detailed information or answer.
    Note: the aim to mimic kv cache memory creation
    Now, the article begins:
    {document}

    The article ends here.

    RESPONSE FORMAT IS IMPORTANT. You must return a JSON object with the following structure:
    {{
        "pairs": [
            {{
                "topic": "topic1",
                "details": "details1"
            }},
            {{
                "topic": "topic2",
                "details": "details2"
            }}
        ]
    }}
    """

system_prompt_for_queries_creation = """You are an expert in information retrieval. 
    Your task is to extract specific and relevant information from the provided context to help answer the given question.
""".strip()

memorag_span_prompt_for_queries_creation = """
    You are given a question related to an article. To answer it effectively, you need to use specific details from the article. You are not provided with the whole article. Instead, you are provided with specific and relevant information from it.
     Your task is to identify and extract one or more specific clue texts from the provided information that are relevant to the question.

    ### Question: {question}
    ### Information: {relevant_info}
    ### Instructions:
    1. You have a general understanding of the provided information. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
    2. The clues are in the form of text spans that will assist in answering the question.
    3. Only output the clues. If there are multiple clues, separate them with a newline.
    """
memorag_sur_prompt_for_queries_creation = """
    You are given a question related to an article. To answer it effectively, you need to use specific details from the article.  You are not provided with the whole article. Instead, you are provided with specific and relevant information from it.
    Your task is to generate precise clue questions that can help locate the necessary information for answering the question.

    ### Question: {question}
    ### Information: {relevant_info}
    ### Instructions:
    1. You have a general understanding of the provided information. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
    2. The clues are in the form of precise surrogate questions that clarify the original question.
    3. Only output the clues. If there are multiple clues, separate them with a newline.
    """

class MemoryStore:
    """The MemoryStore class is a realization of the Memory Module discussed in the paper.
    Its 'memorize' method is used to create a prompt cache mimicking a key-value compression of the original text (database).
    This cache can then be used by the 'create_retrieval_prompt' method for creating the processed query to be used later for retrieval"""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.store = None
        self.processed_count = 0
        self.json_parse_failures = 0
        self._last_parse_used_fallback = False




    def _parse_text_into_used_fallback(self, text:str):
        """Original text parsing methob as fallback.
            Use when LLm didn't return valid json    
        """
        lines = text.split('\n')
        pairs = []
        current_topic = None
        current_detail = []

        for line in lines:
            if line.startswith("Topic:"):
                if current_topic:
                    pairs.append((current_topic," ".join(current_detail)))
                    current_topic.append(line[6:].strip)
                    current_detail=[]
            elif line.startswith("Details:"):
                current_detail.append(line[8:].strip())
            else:
                current_detail.append(line.strip())

        if current_topic:
            pairs.append((current_topic, " ".join(current_detail)))

        return pairs



    def _parse_into_pairs(self, content: str):
        """Use LLM to return the valid Json"""
        self._last_parse_used_fallback = False
        try:
            # Parse JSON and validate with Pydantic
            data = json.loads(content)
            validated_data = MemoryResponse(**data)
            return [(pair.topic, pair.details) for pair in validated_data.pairs]
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            # Fall back to text parsing if JSON fails
            self._last_parse_used_fallback = True
            return self._parse_text_into_pairs(content)


    def memorize(self, document:str):
        """Process document into key-value pairs and store them"""
        system_prompt = memory_store_system_prompt
        kv_cache_prompt = memory_store_kv_cache_prompt

        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {
                    "role":"system", "content": system_prompt
                 },
                 {
                     "role":"user","content": kv_cache_prompt.format(document=document)
                 }
            ],
            response_format = {"type":"json_object"},
            seed = 42
        )
        pairs = self._parse_into_pairs(response.choices[0].message.content)
        if hasattr(self, "json_parse_failures") and self._last_parse_used_fallback:
            self.json_parse_failures += 1
            if self.json_parse_failures >0 and self.json_parse_failures%5 ==0:
                print(f"Warning: JSON parsing has failed {self.json_parse_failures} times")
        print('*'*20)
        print(f'first extracted pairs: {pairs[0:3]}\n\n')

        texts = []
        metadatas = []

        for topic, detail in pairs:
            if not topic or not detail:
                continue

            combine_text = f"Topic: {topic}\nDetails: {detail}"
            texts.append(combine_text)
            metadatas.append({"topic":topic})
        print('*'*20)
        print(f'num of extracted texts: {len(texts)}\n\n')

        if self.store is None:
            self.store = FAISS.from_texts(texts, self.embeddings, metadatas = metadatas)
        else:
            existing_docs = self.store.similarity_search("")
            existing_topics = {doc.metadata.get('topic') for doc in existing_docs}

            # Filter out the dulicates
            new_texts = []
            new_metadatas = []

            for text, metadata in zip(texts, metadatas):
                if metadata["topic"] not in existing_topics:
                    new_texts.append(text)
                    new_metadatas.append(metadatas)

            if new_texts:
                self.store.add_texts(new_texts, metadatas = new_metadatas)

        self.processed_count+=1
        print(f"Processed {self.processed_count} chunks so far\n\n")


    def create_retrieval_queries(self, query: str)-> str | None:
        """
            Generate staging query answers y = Θ_mem(q, D | θ_mem) -> the memorite module produce an output y base on the query q, and data D, using parameter  θ_mem
            This should provide rought clues/outline this guide context retrieval
        """
        if self.store is None:
            return None

        # consider increasing k to more than 10, maybe even top 30% or something
        
        
        results = self.store.similarity_search_with_score(query, k=min(10, self.store.index.ntotal))
        # print(type(results))
        # print(np.array(results).shape)
        
        # relevant_info = [
        #     f"Topic: {doc.metadata['topic']}\nDetails: {doc.page_content}"
        #     for doc, _ in results
        # ]
        relevant_info = [
            f"Topic: {doc.metadata['topic']}\nDetails: {doc.page_content}" for doc,_ in results
        ]

        memorag_sur_prompt = memorag_sur_prompt_for_queries_creation
        memorag_span_prompt = memorag_span_prompt_for_queries_creation
        system_prompt = system_prompt_for_queries_creation

        text_spans = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages =[
                {"role":"system", "content":system_prompt},
                {"role":"user","content":memorag_span_prompt.format(question= query, relevant_info = relevant_info )}
            ]
        ).choices[0].message.content
        surrorate_queries = client.chat.completions.create(
            model = 'gpt-4o-mini',
            messages =[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": memorag_sur_prompt.format(question = query, relevant_info = relevant_info)}
            ],
            temperature = 0.5
        ).choices[0].message.content

        retrieval_query = text_spans.split('\n') + surrorate_queries.split("\n")
        retrieval_query = [q for q in retrieval_query if len(q.strip())>3]
        retrieval_query.append(query)
        return retrieval_query
    def save_store(self, path: str):
        if self.store:
            self.store.save_local(path)

    def load_store(self, path: str):
        self.store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)


"""
Retrieval Function
"""
def retrieval_function(retrieval_query: str, vectorstore, k: int = 3) -> List[str]:
    """
    retrieval relevant context using the staging answer and database vectorstore
    """
    result = vectorstore.similarity_search(retrieval_query, k = k)
    context = [doc.page_content for doc in result]
    return context


"""
Generate function
"""

def generate_function(query: str, context:List[str], temperature : float = 0)-> str:
    """
    Generate final answer 
    """
    prompt = f"""Base ONLY on the provide context, answer the query.
    
    Query: {query}

    Context: 
    {' '.join(context)}

    Provide a clear and concise answer based solely on the information in the context. If the context doesn't contain sufficient information to fully answer the query, clearly state that the necessary information is not available in the provided context rather than using any external knowledge.
"""
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role":"system", "content": "You are an expert in generating answer from given text. Provide clear, concise answers."},
            {"role":"system", "content": prompt}
        ],
        temperature = temperature,
        max_tokens = 500
    )
    return response.choices[0].message.content


def process_query(query: str, memory_store, vectorstore):
    print('*'*20)
    print("\nProcessing Query:", query)
    print("=" * 50)

    # y = Θ_mem(q, D | θ_mem)
    retrieval_queries = memory_store.create_retrieval_queries(query)
    print('*'*20)
    print(f"\n\nRetrieval Queries:\n{retrieval_queries}\n\n")

    # c = Γ(y, D | γ)
    all_contexts = []
    for retrieval_query in retrieval_queries:
        contexts = retrieval_function(retrieval_query, vectorstore, k=3)
        all_contexts.extend(contexts)

    unique_contexts = list(dict.fromkeys(all_contexts))  # Preserves order
    print('*'*20)
    print(f"Retrieved Context Example: {unique_contexts[0]}\n\n")

    # y = Θ(q, c | θ)
    final_answer = generate_function(query, unique_contexts)
    print('*'*20)
    print(f"\nFinal Answer: {final_answer}")

    return unique_contexts, final_answer




"""
MemoRag Short Context


"""

from evaluation.evalute_rag import *

def evaluate_memo_rag(memory_store = None,vectorstore = None, num_questions: int = 5,
                      q_a_file_name = "../data/q_a.json",
                      use_saved_results=True,
                      use_original_memorag=False,
                      original_memorag_answers=None,
                      original_memorag_retrievals=None
                      )-> None:

    """
    Evaluate the MemoRAG system using predefined metrics.

    Args:
        memory_store: MemoryStore object containing the memory model.
        vectorstore: FAISS vector store containing the encoded document chunks.
        num_questions (int): Number of questions to evaluate (default: 5).
        q_a_file_name (str): Path to the JSON file containing questions and answers (default: "../data/q_a.json").
        use_saved_results (bool): Whether to load previously saved results or to save generated results to disk.
        use_original_memorag (bool): Whether to use paper's MemoRAG results or simplified MemoRAG implementation from this notebook.
        original_memorag_answers: Pre-generated answers to use in case use_original_memorag=True.
        original_memorag_retrievals: Pre-generated retrievals to use in case use_original_memorag=True.
    """
    q_a_file_name = q_a_file_name
    with open(q_a_file_name, "r", encoding="utf-8") as json_file:
        q_a = json.load(json_file)

    questions = [qa["question"] for qa in q_a][:num_questions]
    ground_truth_answers = [qa["answer"] for qa in q_a][:num_questions]

    if use_original_memorag:
        """Results of the original MemoRAG were created externally and loaded here"""
        generated_answers = original_memorag_answers
        retrieved_documents = original_memorag_retrievals
        print("Using original MemoRAG results")
    else:
        """When evaluating results of the MemoRAG implementation in this notebook
        there are 2 options - either using pre-saved results or generating new ones
        full_results_path is used either to load pre-saved results or for saving
        the results in case of generation"""
        q_a_file_base = os.path.splitext(q_a_file_name)[0]
        results_path_suffix = "saved_evaluation_results"
        full_results_path = f"{q_a_file_base}_{results_path_suffix}.json"

        if use_saved_results and os.path.exists(full_results_path):
            with open(full_results_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                generated_answers = saved_data["generated_answers"]
                retrieved_documents = saved_data["retrieved_documents"]
            print(f"Loaded previously saved results from {full_results_path}")
        else:
            if memory_store is None or vectorstore is None:
                raise ValueError("memory_store and vectorstore are required when not using saved results or original MemoRAG")

            # Generate answers and retrieve documents for each question
            generated_answers = []
            retrieved_documents = []
            for question in questions:
                contexts, result = process_query(question, memory_store, vectorstore)
                retrieved_documents.append(contexts)
                generated_answers.append(result)

            with open(full_results_path, "w", encoding="utf-8") as f:
                json.dump({
                    "generated_answers": generated_answers,
                    "retrieved_documents": retrieved_documents
                }, f, ensure_ascii=False, indent=2)
            print(f"Saved results to {full_results_path}")

    # Create test cases and evaluate
    test_cases = create_deep_eval_test_cases(questions, ground_truth_answers, generated_answers, retrieved_documents)
    evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric, faithfulness_metric, relevance_metric],
        throttle_value = 5,
        run_async = False,
        ignore_errors = True
    )

original_memorag_answers_wealth = ["Adam Smith",
"The invisible hand refers to the self-interested actions of individuals in a free market economy that unintentionally lead to economic outcomes.",
"Land, labor, and capital.",
"Labor determines the value of goods through the quantity of labor required to produce them.",
"Adam Smith argues that while individuals act in their self-interest, this pursuit often benefits society more effectively than if they acted solely for personal gain. He illustrates this through examples like trade, where individuals acting in their self-interest actually facilitate economic growth and prosperity for society as a whole.",
"Smith justifies taxation as necessary for the defense of the state, the maintenance of public order, and the provision of public services.",
"Supply and demand determine prices. When supply exceeds demand, prices tend to fall; when demand exceeds supply, prices tend to rise.",
"The division of labor encourages workers to focus on specific tasks, leading to innovation in tools and methods. This innovation enhances productivity and has long-term economic impacts by increasing the quantity of goods produced and the efficiency of production processes.",
"Smith viewed agriculture as the primary source of wealth, while manufacturing contributed less significantly.",
"Foreign trade enhances domestic wealth by allowing countries to specialize in producing goods where they have a comparative advantage. This specialization leads to increased efficiency and productivity. Smith's 'invisible hand' refers to the self-interested actions of individuals and businesses that, despite their self-interest, ultimately contribute to economic growth and efficiency.",
"Capital accumulation increases the quantity of productive labor by enabling more workers to be employed. This leads to a division of labor where each worker specializes in a specific task, enhancing efficiency and productivity.",
"Challenges include the need for security, the complexity of international transactions, and the potential for fraud and corruption.",
"Value is the amount of labor required to produce a good or service.",
"Smith believed that government should not interfere with the natural division of labor, allowing individuals to specialize in their areas of expertise. He argued that government's role is to protect property rights, ensure justice, and provide public works and institutions that facilitate economic growth.",
"Free trade benefits consumers by lowering prices, increasing product variety, and enhancing economic efficiency.",
"Adam Smith argued that monopolies, whether natural or artificial, are inherently detrimental to society and the economy. They stifle competition, lead to higher prices, and reduce overall economic efficiency. Smith believed that monopolies are a form of corruption and that the benefits they confer on a few at the expense of many are unjust. He advocated for free trade and competition as essential to economic prosperity and social justice"
]


if __name__ == "__main__":
    # Initialize vector store
    path = "data/Understanding_Climate_Change.pdf"
    # climate_vectorstore = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

    # climate_memory_store = MemoryStore()
    # loader = PyPDFLoader(path)
    # documents = loader.load()
    # document_text = '\n'.join([doc.page_content for doc in documents])
    # climate_memory_store.memorize(document_text)
    # climate_memory_store.save_store("data/Understanding_Climate_Change_Memory_Store.faiss")
    # # # Initialize memory store
    
    # climate_memory_store.load_store("data/Understanding_Climate_Change_Memory_Store.faiss")
    # query_1 = "What are the impacts of climate change on biodiversity?"
    # query_2 = "Please summarize the climate change article"
    # query_3 = "Describe the social and economic influence of climate change."

    # for query in [query_1, query_2, query_3]:
    #     process_query(query, climate_memory_store, climate_vectorstore)


    wealth_vectorstore = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
    
    # Initialize memory store
    wealth_memory_store = MemoryStore()
    wealth_memory_store.load_store("data/The_Wealth_of_Nations_Project_Gutenberg_Memory_Store_Long_Context.faiss")
    
    #Simple RAG Long Context
    chunks_query_retriever = wealth_vectorstore.as_retriever(search_kwargs={"k": 2})

    evaluate_rag(chunks_query_retriever, num_questions = 15, q_a_file_name = "data/q_a_smith_short.json")    
    evaluate_memo_rag(num_questions=15, q_a_file_name = "data/q_a_smith_short.json", memory_store= wealth_memory_store, vectorstore = wealth_vectorstore, use_saved_results=False)

    """





    evaluate_memo_rag(q_a_file_name = "../data/q_a.json", memory_store=climate_memory_store, vectorstore = climate_vectorstore, use_saved_results=False)
    # Initialize vector store
    path = "../data/The_Wealth_of_Nations_Project_Gutenberg.pdf"
    



    #MemoRAG Long Context
    q_a_file_name = "../data/q_a_smith_short.json", memory_store= wealth_memory_store, vectorstore = wealth_vectorstore, use_saved_results=False)

    csv_path = "../data/wealth_retrieved_passages_from_original_memorag.csv"
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    # Convert to a 2D list
    original_memorag_retrievals_wealth = df.values.tolist()
    evaluate_memo_rag(num_questions=15, q_a_file_name = "../data/q_a_smith_short.json", memory_store= wealth_memory_store,
      vectorstore = wealth_vectorstore, use_saved_results=True, use_original_memorag=True, 
      original_memorag_answers=original_memorag_answers_wealth, 
      original_memorag_retrievals=original_memorag_retrievals_wealth)
    """