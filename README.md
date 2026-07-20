# Deep_RAG_system

## Introduction
Retrieval-Augmented Generation (RAG) is revolutionizing the way I combine information retrieval with generative AI. This repository showcases a curated collection of advanced techniques designed to supercharge your RAG systems, enabling them to deliver more accurate, contextually relevant, and comprehensive responses.

## 🌱 Foundational RAG Techniques

### 1. Simple RAG 🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/RAG_simple.py) 

   Start with basic retrieval queries and integrate incremental learning mechanisms.

### 2. Simple RAG using a CSV file 🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/simple_csv_rag.py) 
   This uses CSV files to create basic retrieval and integrates with openai to create question and answering system.

### 3. Reliable RAG 🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/reliable_rag.py) 
Enhances the Simple RAG by adding validation and refinement to ensure the accuracy and relevance of retrieved information.

   #### Implementation 🛠️
   Enhances the Simple RAG by adding validation and refinement to ensure the accuracy and relevance of retrieved information.

### 4. eval RAG with llama_idex🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/llama_index_matric_eval.py) 

   Define function to calculate average response time, average faithfulness and average relevancy metrics for given chunk size.


## 🌱 Query Enhancement

### 1. Query transformation🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Query_enhance/query_transfomation.py) 

   This code implements three query transformation techniques to enhance the retrieval process in Retrieval-Augmented Generation (RAG) systems:
      Query Rewriting
      Step-back Prompting
      Sub-query Decomposition

### 2. Hypothetical Document Embedding (HyDE)🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Query_enhance/HyDE_RAG.py) 

   This code implements a Hypothetical Document Embedding (HyDE) system for document retrieval. HyDE is an innovative approach that transforms query questions into hypothetical documents containing the answer, aiming to bridge the gap between query and document distributions in vector space.

   By leveragin advanced language model to expan queries into hypothetical document. HyDE has potential to significantly improve retriever relevance,
especial for complex or nuanced queries. This techniquie could be practicalarly valuable domain where understanding query intent and context is curcial.
such as  legal research, academic literature review, or advanced information retrieval systems.


### 3. Hypothetical Prompt Embeddings (HyPE)🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Query_enhance/HyPE_RAG.py) 

   HyPE precomputes hypothetical questions during the indexing phase. This transforms retrieval into a question-question matching problem, eliminating the need for expensive runtime query expansion techniques.




## 📚 Context_enrichment

### 1. Contextual Chunk Headers (CCH) 📚[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/CCH_RAG.py) 

   The idea here is to add header in hight level context to the chunk by prepending a chunk header. This chunk header could be as simple as the document title, or it could use a combination of document title, concise document summary, ans the hierarchy of section  and sub-section titles

### 2. Relevant Segment Extraction (RSE)📚[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/RSE_RAG.py) 

      This is method of reconstructing multi-chunk segments  of contiguous text out of retriever chunks.This steps occus after vectorstore ( and optional reranking) but before presenting the retriever contest to the LLM.

      This method ensure that nearby chunks are presented to the LLM in the other they appear in the original document. It also add in chunks that are not marked as relevant, but are sandwitched between  hightly relevant chunks, further improving  the context provided to the LLM.



### 3. Context Window Enhancement 📚[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/Context_window.py) 

      Traditional vector search often returns isolated chunks of text, which may lack necessary context for full understanding. 
      This approach aims to provide a more comprehensive view of the retrieved information by including neighboring text chunks.


### 4. Semantic Chunking 📚[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/segmatic_chunk.py) 

      Semantic chunking represents an advanced approach to document processing for retrieval systems. 
      By attempting to maintain semantic coherence within text segments, it has the potential to improve the quality of retrieved information and enhance the performance of downstream NLP tasks. 
      This technique is particularly valuable for processing long, complex documents where maintaining context is crucial, such as scientific papers, legal documents, or comprehensive reports.

### 5. Contextual Compression 📚[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/compression.py) 

      This code demonstrates the implementation of contextual compression in a document retrieval system using LangChain and OpenAI's language models. 
      The technique aims to improve the relevance and conciseness of retrieved information by compressing and extracting the most pertinent parts of documents in the context of a given query.