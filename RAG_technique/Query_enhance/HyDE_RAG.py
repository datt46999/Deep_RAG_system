"""
Key components:
1. pdf processing and text chunking
2. Vector create using FAISS and Openai Embedding
3. Language model for generating hypothetical documents
4. custom HyDERetriever class implementing th HyDE technique


Retrieval process:
the HyDERetriever process following steps:
1. Generate the hypothetical from user query using llm model.
2. Use the hypothetical document as the  search query in the vectorstore
3. Retriever most similar documents to this hypothetical document


By leveragin advanced language model to expan queries into hypothetical document. HyDE has potential to significantly improve retriever relevance,
especial for complex or nuanced queries. This techniquie could be practicalarly valuable domain where understanding query intent and context is curcial.
such as  legal research, academic literature review, or advanced information retrieval systems.


user_query --> generate hypothetical Document --> Embedding  hypothetical document --> similary search in Docuement store -->return relavant documents

                                                                                             ^
                                                                        Document store ------|
"""


import os 
from dotenv import load_dotenv

from helper_functions import encode_pdf, show_context, text_wrap
from langchain_openai import ChatOpenAI, OpenAIEmbedding
from langchain_core.prompts import PromptTemplate


load_dotenv()

# from langchin_text_splitters import RecursiveCharacterTextSplitter 
# from langchain_community.loader_documents import PyPDFLoader
# from langchain_core.vectorstores import FAISS


# def encode_pdf(path, chunk_size = 1000, overlap = 200):
#     loader = PyPDFLoader(path)
#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size =  1000,
#         chunk_overlap = 200,
#         length_function = len
#     )
#     text_splitter = splitter.split_document(docs)
#     embedding = OpenAIEmbedding()
#     vectorstore = FAISS.from_document(text_splitter, embedding)
#     return vectorstore

    
    

class HyDERetriever:
    def __init__(self, data_path, chunk_size = 1000, chunk_overlap = 200):
        self.llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0, max_tokens = 4000)
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(data_path, chunk_size= chunk_size, chunk_overlap= chunk_overlap)

        self.hyde_prompt = PromptTemplate(
            input_variables = ["query", "chunk_size"],
            template = """Given the question '{query}', generate a hypothetical document that directly answer this quesion. Document should be detailed and in-depth,
            the document size have exactly {chunk_size} characters.
            """
        )

        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetic(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content
    def retrieve(self, query, k = 3):
        hypothetical_doc = self.generate_hypothetic(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k = k)
        return similar_docs, hypothetical_doc
    
if __name__ == "__main__":
    path = 'data/Understanding_Climate_Change.pdf'
    retriever = HyDERetriever(path)

    text_query = "What is main cause when climate change?"
    results, hypothetical_doc = retriever.retrieve(text_query)

    doc_content = [doc.page_content for doc in results]

    print("hypothetical_doc:\n")
    print(text_wrap(hypothetical_doc)+"\n")
    show_context(docs_content)