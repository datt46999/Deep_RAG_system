"""
Implement basic RAG for processing and query with csv document
Using Faiss and open ai embedding
"""





import pandas as pd
import os

from dotenv import load_dotenv


from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import faiss

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


system_prompt = ("""
    You are assistant for question-answerin task.
    Use the following piece of retrieved context to answer the question.
    If don't have answer, say that  you don't know. Use three sentence maximum and keep the answer  concise.
    \n\n
    {context}
""")
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', "{input}"),
    ]
)

def embedding2vectore(data):
    loader = CSVLoader(data)
    docs = loader.load()

    embedding = OpenAIEmbeddings()
    index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))
    vectorstore = FAISS(
        embedding_function=OpenAIEmbeddings(),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vectorstore.add_documents(documents = docs)
    return vectorstore




if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini")


    file_path = "data/customers-100.csv"
    # data = pd.read_csv(file_path)
    # print(data.head())
    vectorstore = embedding2vectore(file_path)
    retriever = vectorstore.as_retriever()

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    answer= rag_chain.invoke({"input": "which company does sheryl Baxter work for?"})
    print(answer["answer"])
