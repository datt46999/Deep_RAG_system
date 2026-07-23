# Deep_RAG_system

## Introduction
Retrieval-Augmented Generation (RAG) is revolutionizing the way we combine information retrieval with generative AI. This repository showcases a curated collection of advanced techniques designed to supercharge your RAG systems, enabling them to deliver more accurate, contextually relevant, and comprehensive responses.

---
## 📌 Quick Directory Index

* [🌱 Foundational RAG Techniques](#-foundational-rag-techniques) (4 files)
* [🚀 Query Enhancement](#-query-enhancement) (3 files)
* [📚 Context Enrichment](#-context-enrichment) (6 files)
* [⚡ Advanced Retrieval](#-advanced-retrieval) (5 files)
* [🎯 Advanced RAG Architectures](#-advanced-rag-architectures) (5 files)

---

## 🌱 Foundational RAG Techniques

### 1. Simple RAG 🌱 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/RAG_simple.py) 
Implements basic retrieval queries and integrates incremental learning mechanisms to form the backbone of a RAG pipeline.

### 2. Simple RAG using a CSV file 🌱 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/simple_csv_rag.py) 
Utilizes CSV files as structured data sources to perform basic text retrieval and integrates with OpenAI to build an efficient question-answering system.

### 3. Reliable RAG 🌱 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/reliable_rag.py) 
Enhances the standard baseline by adding automated validation and refinement loops, ensuring the strict accuracy and relevance of retrieved information before generation.

### 4. Eval RAG with LlamaIndex 🌱 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/llama_index_matric_eval.py) 
Defines specialized evaluation functions using LlamaIndex to measure and optimize average response time, faithfulness, and relevancy metrics across varying chunk sizes.

---

## 🚀 Query Enhancement

### 1. Query Transformation 🌱 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Query_enhance/query_transfomation.py) 
Implements three sophisticated optimization techniques to bridge the gap between user intent and document retrieval:
* **Query Rewriting:** Rephrasing the input for clarity.
* **Step-back Prompting:** Abstracting the query to capture high-level conceptual context.
* **Sub-query Decomposition:** Breaking down complex queries into manageable sub-questions.

### 2. Hypothetical Document Embedding (HyDE) 🌱 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Query_enhance/HyDE_RAG.py) 
An innovative approach that uses an LLM to generate a hypothetical answer document from the query first, using that hallucinated document to search the vector space. This aligns query and document distributions, making it highly valuable for complex domain searches like legal or academic literature.

### 3. Hypothetical Prompt Embeddings (HyPE) 🌱 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Query_enhance/HyPE_RAG.py) 
Precomputes hypothetical questions during the offline indexing phase. This transforms runtime retrieval into a direct question-to-question matching problem, significantly saving latency by eliminating the need for expensive online query expansion.

---

## 📚 Context Enrichment

### 1. Contextual Chunk Headers (CCH) 📚 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/CCH_RAG.py) 
Prevents the loss of global context by prepending high-level metadata headers to isolated chunks. These headers can include document titles, concise summaries, or hierarchical section maps.

### 2. Relevant Segment Extraction (RSE) 📚 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/RSE_RAG.py) 
Reconstructs multi-chunk, contiguous text blocks post-retrieval. RSE ensures nearby chunks are presented in their original chronological order and automatically fills in unindexed "sandwich" chunks to provide seamless context to the LLM.

### 3. Context Window Enhancement 📚 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/Context_window.py) 
Overcomes the limitations of isolated text chunks by automatically pulling in adjacent sliding window contexts (surrounding text blocks) to offer a more comprehensive view during generation.

### 4. Semantic Chunking 📚 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/segmatic_chunk.py) 
An advanced text processing approach that splits documents based on semantic transitions rather than arbitrary character limits. This maintains structural coherence—ideal for long, dense texts like scientific papers or financial reports.

### 5. Contextual Compression 📚 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/compression.py) 
Utilizes LangChain and OpenAI to compress and extract only the most pertinent sentences from retrieved documents. By stripping out irrelevant noise before passing context to the prompt, it reduces costs and minimizes LLM distraction.

### 6. Augmentation technique 📚 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Context_enrichment/document_augmentation.py) 
This implementation demonstrates a text augmentation technique the leverages additional question generate to improve document retriever within a vector database.

---
## 🚀 Advance Retrieval

### 1. Fusion retrieval 🚀 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Advance_RAG/fusion_retrieval.py) 
Fusion retrieval aims to combine these methods to create a more robust and accurate retrieval system that can handle a wider range of queries effectively.
### 2. Multimodel retrieval 🚀 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Advance_RAG/multimodel.py) 
Extracts and processes text and images from PDFs,
utilizing a multi-modal Retrieval-Augmented Generation (RAG) system for
summarizing and retrieving content for question answering.

### 3. Reranking 🚀 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Advance_RAG/Reranking.py) 
Reranking in RAG systems is to overcome limitations of initial retrieval methods, which often rely on simpler similarity metrics. 

### 4. Hierachical retrieval 🚀 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Advance_RAG/hierarchical.py) 
Implements a Hierarchical index system for document retrieval, utilizing two level for encoding: document level summaries and detail chunking.

### 5. DartBoard retrieval 🚀 [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/Advance_RAG/dartboard.py) 
The DartBoard RAG process addresses a common challenge in large knowledge base: ensuring the retrieved information in both relevant and no-relevant.
By explicitly optimizing a combined relevance-diversity scoring function, it prevents top-k documents from offering same information.


