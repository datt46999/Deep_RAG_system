"""
This implementation demonstrates a text augmentation technique the leverages additional question generate to improve document retriever within a vector database.
By generating and incorporating  various question related of each text fragment,this system enhance retrievel process, 
thus increasing the likelihood of of finding  relevant documents that can be utilized as context for generative question answering


Key components:
1.  PDF processing and text chunking:
2. Question Augmentation: generating relevant question as both document and fragment level using Open Ai
2. Vectorstore creation: Calculating embeddings for documents using OpenAI's embedding model and creating a FAISS vector store.
4. Retrieval and answer generate: Finding the most relevant document using FAISS and generating answers based on the context provided.
"""

import os
import fitz
import re
from typing import Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from enum import Enum
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings




load_dotenv()

class QuestionGeneration(Enum):
    """
    Enum class to specify the level of question generation for document processing.

    Attributes:
        DOCUMENT_LEVEL (int): Represents question generation at the entire document level.
        FRAGMENT_LEVEL (int): Represents question generation at the individual text fragment level.
    """
    DOCUMENT_LEVEL = 1
    FRAGMENT_LEVEL = 2
class QuestionList(BaseModel):
    questions_list: list[str] = Field(...,titles ='List of question generated for document or fragment' )

#Depending on the model, for Mitral 7B it can be max 8000, for Llama 3.1 8B 128k
DOCUMENT_MAX_TOKENS = 4000
DOCUMENT_OVERLAP_TOKENS = 100

#Embeddings and text similarity calculated on shorter texts
FRAGMENT_MAX_TOKENS = 128
FRAGMENT_OVERLAP_TOKENS = 16

#Questions generated on document or fragment level
QUESTION_GENERATION = QuestionGeneration.DOCUMENT_LEVEL
#how many questions will be generated for specific document or fragment
QUESTIONS_PER_DOCUMENT = 10

class OpenAI_Embedding_Wrapper(OpenAIEmbeddings):
    """
    A Wrapper class for  OpenAI embedding, providing a similar interface to the original allamaEmbeddings

    """
    def __call__(self, query:str)-> list[float]:
        """
        Allows the instance to be used as a callable to generable  an embedding for a query
        args:
        query(str): The query string to be embedding
        return:
        List[float]: the embedding for the query as a list of float.
        """
        return self.embed_query(query)
def read_pdf_to_string(path):
    """
    Read document pdf in speciafized path and return content as a string
    
    """
    docs =  fitz.open(path)
    content = ""
    for page_num in range(len(docs)):
        page = docs[page_num]
        content+= page.get_text()

    return content


def clear_and_filter_questions(questions: list[str])->list[str]:
    """
    clear and filter list of question
    arg:
    question list[str]: a list of question to be cleaned and filtered
    return:
    List[str]: a list of cleaned and filtered questions that end with a question mark.

    """
    clean_questions = []
    for question in questions:
        clean_question = re.sub(r'^\d+\.\s*', '', question.strip())
        if clean_question.endswith('?'):
            clean_questions.append(clean_question)
    return clean_questions

def generate_question(text: str) -> list[str]:
    """
    generate list of question base on provided text using OpenAI
    ARG:
    text(str): the context data from which questions are generate
    return:
    list[str]:list of unique, filtered quesions
    """
    llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
    prompt = PromptTemplate(
        input_variables =['context', 'num_questions'],
        template = """Using the context data: {context}\n\n Generate a list at least{num_questions}possible questions that can be ask about this context.
                        Ensure the questions are directly answerable within context and do not include  any answers or headers.
                        Separate questions with a new line character.
            """
    )
    chain = prompt | llm.with_structured_output(QuestionList)
    
    input_data = {"context": text, "num_questions": QUESTIONS_PER_DOCUMENT}
    generate_questions = chain.invoke(input_data)
    questions = generate_questions.questions_list
    filter_questions = clear_and_filter_questions(questions)
    return list(set(filter_questions))

def generate_answer(content: str, question: str) -> str:
    """
    Generate the answer to a given question base on the provided context using OpenAI
    ARG:
    content(str):The context data use for generate the answer
    questions(str):The question for which the answer  is generate

    return:
    str: the precise answer to the question bace in provided context
    """
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    prompt = PromptTemplate(
        input_variables =['context', 'question'],
        template = "Using the context data: {context}\n\n Provide a brief and specise answer to the question: {question}"
    )
    chain = prompt | llm
    
    input_data = {"context": content, "question": question}
    return chain.invoke(input_data)


def split_document(document: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split document into smaller chunks of text
    ARG:
    document (str): text of document to be split
    chunk_size (int): the sizes of each chunk in terms of the number of token
    chunk_overlap (int): the number of overplapping tokens between consecutive chunks.

    return 
    List[str]: A list of text chunks, where each chunk is a string of the document content. 
    """
    token = re.findall(r'\b\w+\b', document)
    chunks = []
    for i in range(0, len(token), chunk_size- chunk_overlap):
        chunk_token = token[i:i + chunk_size]
        chunks.append(chunk_token)
        if i+ chunk_size>= len(token):
            break
    return [" ".join(chunk) for chunk in chunks]





def print_document(comment: str, document: Any) -> None:
    """
    Prints a comment followed by the content of a document.

    Args:
        comment (str): The comment or description to print before the document details.
        document (Any): The document whose content is to be printed.

    Returns:
        None
    """
    print(f'{comment} (type: {document.metadata["type"]}, index: {document.metadata["index"]}): {document.page_content}')



def process_documents(content: str, embedding_model: OpenAIEmbeddings):
    """
    Process documents context, split it into framents, generate questions,
    create FAISS vector store and return retrievel
    ARG:
    content(str): The content of document process
    embedding_model: the embedding model to use for vectorstore

    """

    text_documents = split_document(content, DOCUMENT_MAX_TOKENS, DOCUMENT_OVERLAP_TOKENS)
    print(f'Text content split into: {len(text_documents)} documents')
    documents = []
    counter = 0
    for i, text_document in enumerate(text_documents):
        text_fragments = split_document(content, FRAGMENT_MAX_TOKENS, FRAGMENT_OVERLAP_TOKENS)
        print(f'Text content split into: {len(text_fragments)} documents')

        for j, text_fragment in enumerate(text_fragments):
            documents.append(
                Document(
                    page_content = text_fragment,
                    metadata = {'type': "ORIGINAL", 'index': counter,"text": text_document}
                )
            )
            counter +=1


            if QUESTION_GENERATION == QuestionGeneration.FRAGMENT_LEVEL:
                questions = generate_question(text_fragment)
                documents.extend([
                    Document(page_content=question, metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                    for idx, question in enumerate(questions)
                ])
                counter += len(questions)
                print(f'Text document {i} Text fragment {j} - generated: {len(questions)} questions')

        if QUESTION_GENERATION == QuestionGeneration.DOCUMENT_LEVEL:
            questions = generate_question(text_document)
            documents.extend([
                Document(page_content=question, metadata={"type": "AUGMENTED", "index": counter + idx, "text": text_document})
                for idx, question in enumerate(questions)
            ])
            counter += len(questions)
            # print(f'Text document {i} - generated: {len(questions)} questions')
# 
    # for document in documents:
    #     print_document("Dataset", document)

    print(f'Creating store, calculating embeddings for {len(documents)} FAISS documents')
    vectorstore = FAISS.from_documents(documents, embedding_model)

    print("Creating retriever returning the most relevant FAISS document")
    return vectorstore.as_retriever(search_kwargs={"k": 1})


if __name__ =="__main__":
    embeddings = OpenAI_Embedding_Wrapper()
    path = "data/Understanding_Climate_Change.pdf"

    example_text = "This is an example document. It contains information about various topics."
    # questions = generate_question(example_text)
    
    # # generate question
    # print("=="*40)
    # print("Question generate: ")
    # for q in questions:
    #     print(f"-  {q}")
    # print("=="*40)

    # # generate answer 
    # sample_question = questions[0] if questions else "What is this document about?"
    # answer = generate_answer(example_text, sample_question)
    # print(f"\nQuestion: {sample_question}")
    # print(f"Answer: {answer}")

    # print("=="*40)
    # chunks = split_document(example_text, chunk_size=10, chunk_overlap=2)
    # print("\nDocument Chunks:")
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i + 1}: {chunk}")

    # # Example of using OpenAIEmbeddings
    # doc_embedding = embeddings.embed_documents([example_text])
    # query_embedding = embeddings.embed_query("What is the main topic?")
    # print("\nDocument Embedding (first 5 elements):", doc_embedding[0][:5])
    # print("Query Embedding (first 5 elements):", query_embedding[:5])


    content = read_pdf_to_string(path)
    embedding_model = OpenAIEmbeddings()
    document_query_retriever = process_documents(content, embedding_model)
    # Example usage of the retriever
    query = "What is climate change?"
    retrieved_docs = document_query_retriever.invoke(query)
    print(f"\nQuery: {query}")
    print(f"Retrieved document: {retrieved_docs[0].page_content}")
    query = "How do freshwater ecosystems change due to alterations in climatic factors?"
    print (f'Question:{os.linesep}{query}{os.linesep}')
    retrieved_documents = document_query_retriever.invoke(query)

    for doc in retrieved_documents:
        print_document("Relevant fragment retrieved", doc)
        context = doc.metadata['text']
        print (f'{os.linesep}Context:{os.linesep}{context}')
        answer = generate_answer(context, query)
        print(f'{os.linesep}Answer:{os.linesep}{answer}')