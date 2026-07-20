"""
Key components:
1. PDF processing and text extraction
2. Text chunking to maintain coherent information units
3. Hypothetical prompt embedding generation using LLM to create multiple proxy questions per chunk
4. Vector store creation using FAISS and OPENAI embedding
5. Retriever up for query and process documents
6. Evaluation the RAG system




HyPE generate multiple hypothetical prompt for each query. These precomputed questions similate user queries, Improving aligment with real world searches.
this removes the runtime sythetic answer generation needed to techiques like HyDE
this increase  retriever flexible from  approach  store multipler represtations per chunk (association each question embedding with its orginal chunk)

"""



import os
from typing import List
from tqdm import tqdm
import faiss


from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.output_parpers import StrOutoutParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore



from helper_functions import *
from evaluation.evalute_rag import *
load_dotenv()



LLM_MODELS = "gpt-4.0-mini"
MODEL_EMBEDDING = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVELAP = 200
def generate_hype_embedding(chunk_text):
    """
    Uses the LLM to generate multiple hypothetical question for a single chunk.
    These question will be used as 'proxies' for the chunk during retrieval.
    ARG:
    chunk_text (str): Text contents of the chunk

    Returns:
    chunk_text (str): Text contents of the chunk. This is done to make the 
        multithreading easier
    hypothetical prompt embeddings (List[float]): A list of embedding vectors
        generated from the questions
    """

    llm = ChatOpenAI(model = LLM_MODELS, temperature = 0)
    embedding = OpenAIEmbeddings(model = MODEL_EMBEDDING)
    
    quesion_gen_prompt = PromptTemplate.from_tempate(
        """
        Analyze the input text and generate essential question that, when answers,
        capture main point of the text. Each question should be one line, without numbers or prefexis\n\n
        Text:\n{chunk_text}\n{question:}\n
        """
    )

    
    questions_chain = quesion_gen_prompt | llm | StrOutoutParser()
    question = questions_chain.invoke({"chunk_text":chunk_text }).replace("\n\n","\n").split("\n")

    return chunk_text, embedding.embed_documents(question) 



"""
Embedding chunk size with parallel
"""
def prepare_vectorstore(chunks: List[str]):
    """
    Create and Populates a FAISS vector store for list of text chunk.

    This fuction processes a list of text chunk in parallel, 
    generate hypothetical prompt embedding for each chunk.

    The embedding are store in FAISS infex for embedding and store.
     
    ARG:
    chunk: List of text chunk to embedding into vectorstore
    return:
    FAISS vector store
    """

    vectorstore = None
    with ThreadPoolExecutor() as pool:
        # use threading to speed up generation of prompt embedding
        features = [pool.submit(generate_hype_embedding,c) for c in chunks]


        # process embedding as they complete
        for f in tqdm(as_completed(features), total=len(chunks)):
            chunk, vectors = f.result()

            if vectorstore == None:
                vectorstore = FAISS(
                    embedding_function=OpenAIEmbeddings(model=MODEL_EMBEDDING),  # Define embedding model
                    index=faiss.IndexFlatL2(len(vectors[0])),  # Define an L2 index for similarity search
                    docstore=InMemoryDocstore(),  # Use in-memory document storage
                    index_to_docstore_id={}  # Maintain index-to-document mapping
                )
            chunk_with_embedding_vectors = [(chunk.page_content, vec) for vec in vectors]

            vectorstore.add_embeddings(chunk_with_embedding_vectors)
    return vectorstore


def encode_pdf(path, chunk_size = 1000, chunk_overlap = 200):
    """
    Encoder PDF file using OpenAi embedding
    
    """
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )        
    text_splitter = splitter.split_documents(docs)

    
    clear_text = replace_t_space(text_splitter)
    vectorstore = prepare_vectorstore(clear_text)
    return vectorstore

if __name__ == "__main__":
    PATH = "data/Understanding_Climate_Change.pdf"
 
    chunks_vector_store = encode_pdf(PATH)
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 3})


    test_query = "What is the main cause of climate change?"
    context = retrieve_context_per_question(test_query, chunks_query_retriever)
    context = list(set(context))
    show_context(context)


    evaluate_rag(chunks_query_retriever)