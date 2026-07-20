
## 🌱 Foundational RAG Techniques

### 1. Simple RAG 🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/RAG_simple.py) 

   #### Overview 🔎
   Introducing basic RAG techniques ideal for newcomers.

   #### Implementation 🛠️
   Start with basic retrieval queries and integrate incremental learning mechanisms.

### 2. Simple RAG using a CSV file 🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/simple_csv_rag.py) 

   #### Overview 🔎
   Introducing basic RAG techniques using csv file.

   #### Implementation 🛠️
   This uses CSV files to create basic retrieval and integrates with openai to create question and answering system.

### 3. Reliable RAG 🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/reliable_rag.py) 

   #### Overview 🔎
   
   


```text
Start (Query)
      │
      ▼
Vector Store Retrieval
(Query → Retrieve Relevant Documents)
      │
      ▼
Document Relevance Check
(Query + Retrieved Documents)
      │
      ▼
Answer Generation
(Relevant Documents + Query)
      │
      ▼
Hallucination Check
(Query + Relevant Documents + Generated Answer)
      │
      ▼
Highlight Supporting Document Snippets
      │
      ▼
    End


   ```

   #### Implementation 🛠️
   Enhances the Simple RAG by adding validation and refinement to ensure the accuracy and relevance of retrieved information.

### 4. eval RAG with llama_idex🌱[<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/datt46999/Deep_RAG_system/blob/main/RAG_technique/foundation/llama_index_matric_eval.py) 

   #### Overview 🔎
   Relevancy Evaluators which are based on GPT-4
   #### Implementation 🛠️
   Define function to calculate average response time, average faithfulness and average relevancy metrics for given chunk size.

