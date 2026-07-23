"""
Implements one of the multiple ways of multi-modal RAG.
It extracts and processes text and images from PDFs,
utilizing a multi-modal Retrieval-Augmented Generation (RAG) system for
summarizing and retrieving content for question answering.

Key components:
PyMuPDF: For extracting text and images from PDF
Gemini 1.5-flash model: To summarize images and tables
Cohere Embedding: For embedding document splits
Chroma Vectorstore: To store and retrieve document embeddings
LangChain: To orchestrate the retrieval and generation pipeline.
"""

import fitz
import os
import io
from PIL import Image
from dotenv import load_dotenv
 
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
 
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel(model_name="gemini-2.5-flash")
 
 
def summarize_image(image_path):
    """Use Gemini to produce a short text summary of an extracted image,
    so it can be embedded and retrieved alongside the text chunks."""
    image = Image.open(image_path)
    prompt = (
        "Describe this image or table in detail, focusing on any data, "
        "numbers, or facts a reader might search for. Keep it concise."
    )
    response = model.generate_content([prompt, image])
    return response.text.strip()
 
 
def data_extraction(path):
    text_data = []
    img_data = []
    img_dir = "data/extracted_images"
    os.makedirs(img_dir, exist_ok=True)
 
    with fitz.open(path) as file:
        for page_number in range(len(file)):
            page = file[page_number]
 
            # Get the text on the page
            text = page.get_text().strip()
            text_data.append({"response": text, "name": page_number + 1})
 
            # Get the list of images on the page
            images = page.get_images(full=True)
 
            for image_index, img in enumerate(images, start=1):
                xref = img[0]  # XREF of the image
                base_image = file.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
 
                image = Image.open(io.BytesIO(image_bytes))
                img_path = os.path.join(
                    img_dir, f"image_{page_number + 1}_{image_index}.{image_ext}"
                )
                # Guard against save failures on odd color modes (e.g. CMYK)
                try:
                    image.save(img_path)
                except OSError:
                    image.convert("RGB").save(img_path)
 
                # Summarize the image so it has real text content to embed
                summary = summarize_image(img_path)
                img_data.append({"response": summary, "name": page_number + 1})
 
    return text_data, img_data
 
 
def create_vector(text_data, img_data):
    embedding_model = CohereEmbeddings(model="embed-english-v3.0")
 
    # Load the documents
    docs_list = [
        Document(page_content=text["response"], metadata={"name": text["name"]})
        for text in text_data
    ]
    img_list = [
        Document(page_content=img["response"], metadata={"name": img["name"]})
        for img in img_data
    ]
 
    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    img_splits = text_splitter.split_documents(img_list)
 
    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits + img_splits,
        collection_name="multi_model_rag",
        embedding=embedding_model,
    )
 
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    return retriever
 
 
def process(query, retriever):
    system = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge.
    Use three-to-five sentences maximum and keep the answer concise."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>",
            ),
        ]
    )
 
    # Actually run retrieval first
    docs = retriever.invoke(query)
 
    llm = ChatCohere(model="command-a-03-2025", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
 
    generation = rag_chain.invoke(
        {"documents": docs[0].page_content, "question": query}
    )
    print(generation)
 
 
if __name__ == "__main__":
    path = "data/attention_is_all_you_need.pdf"
    text_data, img_data = data_extraction(path)
    retriever = create_vector(text_data, img_data)
 
    query = "What is the BLEU score of the Transformer (base model)?"
 
    process(query, retriever)
 