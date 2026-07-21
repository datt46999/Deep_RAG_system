# Deep_RAG_system

## Introduction
Retrieval-Augmented Generation (RAG) is revolutionizing the way we combine information retrieval with generative AI. This repository showcases a curated collection of advanced techniques designed to supercharge your RAG systems, enabling them to deliver more accurate, contextually relevant, and comprehensive responses.

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
