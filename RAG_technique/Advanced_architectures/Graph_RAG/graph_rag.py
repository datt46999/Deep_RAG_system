"""
Graph RAG is an advanced question-answering system that combines the power of graph base knowledge represetaion with retrieval-argumented generation.
It processes input documents to create a rich knowledge graph, which is then use to enchance the retrieval and generation of answer to user queries.
The system leverages natural language processin, MC, and graph theory to provide more accurace and contextually relevant responses



Methobs details:
1. Document Processing
    Input documents are split into manageable chunks.
    Each chunk are embedding with language model.
    A vector store is create for these embeddings fot efficient similarity search.
2. Knowledge graph construction:
    Graph nodes are created for each text chunk.
    Concepts are extracted from each chunk using a combination of NLP techniques and language models.
    Extracted concepts are lemmatized to improve matching.
    Edges are added between nodes based on semantic similarity and shared concepts.
    Edge weights are calculated to represent the strength of relationships.
3. Query Processing:

    The user query is embedded and used to retrieve relevant documents from the vector store.
    A priority queue is initialized with the nodes corresponding to the most relevant documents.
    The system employs a Dijkstra-like algorithm to traverse the knowledge graph:
        Nodes are explored in order of their priority (strength of connection to the query).
        - For each explored node:
            Its content is added to the context.
            The system checks if the current context provides a complete answer.
            If the answer is incomplete:
            The node's concepts are processed and added to a set of visited concepts.
            Neighboring nodes are explored, with their priorities updated based on edge weights.
            Nodes are added to the priority queue if a stronger connection is found.
            This process continues until a complete answer is found or the priority queue is exhausted.
            If no complete answer is found after traversing the graph, the system generates a final answer using the accumulated context and a large language model.
4. Visualization:

    The knowledge graph is visualized with nodes representing text chunks and edges representing relationships.
    Edge colors indicate the strength of relationships (weights).
    The traversal path taken to answer a query is highlighted with curved, dashed arrows.
    Start and end nodes of the traversal are distinctly colored for easy identification.

"""


import os 
import numpy as np
import networkx as nx 
import heapq
import nltk

from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.callbacks import get_openai_callback

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import spacy 
from spacy.cli import download
from spacy.lang.en import English

from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)



"""
=============================================================
Define the document processor class 
=============================================================
"""
class DocumentProcessor:
    def __init__(self):
        """
        Inititlized with Document Processor with a text splitter and Open AI embedding
        
        Attributes:
         - text_splitter: An instance of RecursiveCharacterTextSplitter with specified chunk size and overlap
         - embeddings: An instance of OpenAI Embedding using embedding documents
        """

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap =200)
        self.embeddings = OpenAIEmbeddings()
    def process_documents(self, documents):
        """
        Processes a list document by splittinnf them into samller chunks and creating vectorstore

        ARG:
            documents (list of str): A List of document to precessed
        returns:
            tuple: A tuple containing:
                - splits (list of str): The list of document chunks
                - vectorstore (FAISS): a FAISS vector store created from the splitter document chunk and their embeddings 
        """
        splits = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, self.embeddings) 
        return splits, vectorstore

    def create_embedding(self, text, batch_size):
        """
        Creates embedding for a list of texts in batch size

        ARG:
            text (list of str): List of test embeddings
            batch_size (int, optional): The number of texts to process in each batch

        Returns:
            numpy.ndarray: an array of embeddings for  the input text  
        """
        embeddings = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    def compute_similarity_matrix(self, embeddings):
        """
        Computes a consine similarity matrix for given set of embedding.
        Args:
        - embeddings (numpy.ndarray): An array of embeddings.
        
        Returns:
        - numpy.ndarray: A cosine similarity matrix for the input embeddings.
        """
        return cosine_similarity(embeddings)


"""
=============================================================
Define the knowledge graph class

=============================================================
"""
class Concepts(BaseModel):
    concept_list : List[str] = Field(description = "List of concepts")




class Knowledge_graph:
    def __init__(self):
        """
        Initialize  the knowledge with the graph, lemmatizer and NLP model
        
        Attributes:
        graph: Instance of a network Graph
        lemmatizer: An instance of WordNetLemmatizer
        concept_cache: A dictionary to cache extracted concepts
        nlp: an instance of a spaCy NLP model
        edges_threshold: A float value that sets the threshold for adding edges based on similarity
        """
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edge_threshold = 0.8

    def build_graph(self, splits, llm, embedding_model):
        """
        Build the knowledge graph by adding nodes, creating embeddings, extracting concept, and adding nodes
        Args:
        splitts : A list of document splits
        llm:  an instance of large language model
        embedding_model: an instanve of embedding model

        return None
        """
        self._add_nodes(splits)
        embedding = self._create_embeddings(splits, embedding_model)
        self._extract_concept(splits, llm)
        self._add_edges(embedding)

    def _add_nodes(self, splits):
        """
        Add nodes to graph from document splits
        ARG:
        splits List[str]: List of document
        return:
        None
        """
        for i, split in enumerate(splits):
            self.graph.add_node(i, content = split.page_content)


    def _create_embeddings(self, splits, embedding_model):
        """
        Create embedding for document splits using large language model
        Args:
            splits List: A list of document splits
            embedding_model: An instance of an embedding model
        Return:
            numpy.ndarray: A cosine similarity matrix
        """

        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)
    def _load_spacy_model(self):
        """
        Load spacy model and download it if necessery
        Args:
            None
        return:
            spacy.language: An instance of a spacy NPL model
        """
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concept_and_entities(self, content, llm):
        """
        Extract concept and name entities using spacy and model LLM

        Args:
            content str: the content from which to extract concepts and entities
            llm: An instance of a llm
        return:
            list: list of extract concept and entities
        """
        # check content have exists
        if content in self.concept_cache:
            return self.concept_cache[content]

        # Extract name entities using spacy
        docs = self.nlp(content)
        name_entities = [entity.text for entity in  docs.ents if entity.label_ in  ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        # Extract generate concept using LLM
        prompt_concept = PromptTemplate(
            input_variables = ["content"],
            template = "Extract the keys concept (excluding named entities) from the following text:\n\n{text}\n\nKey concepts: "
        )
        concept_chain = prompt_concept | llm.with_structured_output(Concepts)
        generate_concepts = concept_chain.invoke({"text": content}).concept_list


        # Conbine named entities and generate concepts
        combine_results = list(set(name_entities + generate_concepts))
        self.concept_cache[content] = combine_results
        return combine_results
    def _extract_concept(self, splits, llm):
        """
        Extract concept for all document split using multi-threading
        Args :
        splits (list): A list of document split
        llm: an instance of a large language model

        return None

        """
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concept_and_entities, split.page_content, llm): i for i, split in enumerate(splits)}
            for future in tqdm(as_completed(future_to_node), total = len(splits), desc = "Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph.nodes[node]["concepts"] = concepts


    def _compute_similarities(self, embeddings):
        """
        Compute the cosine similarity matrix for the the embedding

        Args:
            -embedding (numpy.ndarray): An array of embedding
        return 
            numpy.ndarray: A cosine similarity matrix for the embedding 
        """
        return cosine_similarity(embeddings)

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """
        Calculates the weight of an edge based on similarity score and shared concepts.
        
        Args:
        - node1 (int): The first node.
        - node2 (int): The second node.
        - similarity_score (float): The similarity score between the nodes.
        - shared_concepts (set): The set of shared concepts between the nodes.
        - alpha (float, optional): The weight of the similarity score. Default is 0.7.
        - beta (float, optional): The weight of the shared concepts. Default is 0.3.
        
        Returns:
        - float: The calculated weight of the edge.
        """
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts


    def _add_edges(self, embeddings):
        """
        Add edges to the graph based on the similarity of embedding  and shared concepts

        Args:
            embeddings (numpy.ndarray): An array of embeddings for the document splits
        
        Return:
            None
        """
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodels = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodels), desc = "Adding edges"):
            for node2 in range(node1 +1, num_nodels):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edge_threshold:
                    shared_concepts = set(self.graph.nodes[node1]["concepts"]) & set(self.graph.nodes[node2]["concepts"])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight=edge_weight, 
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))

    def _lemmatize_concept(self, concept):
        """
        Lemmatizes a given concept.
        
        Args:
        - concept (str): The concept to be lemmatized.
        
        Returns:
        - str: The lemmatized concept.
        """
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])


"""
=============================================================
Define the answercheck and QueryEngine class
=============================================================
"""
class AnswerCheck(BaseModel):
    is_complete : bool = Field(description ="Whether the current context provides a complete answer to the query")
    answer : str = Field(description = "The current answer base on context, if any")

# define QueryEngine
class QueryEngine:
    def __init__(self, vectorstore, knowledge_graph, llm):
        self.vectorstore = vectorstore
        self.knowledgegraph = knowledge_graph
        self.llm = llm
        self.max_context_length = 4000
        self.answer_check_chain = self._create_answer_check_chain()
    def _create_answer_check_chain(self):
        """
        Create a chain to check if the query  provides a complete answer to the query

        Args:
            None
        Return:
            Chain: The chain to check if the concext provides a complete answer
        """
        answer_check_prompt = PromptTemplate(
            input_variables = ["query", "context"],
            template = """Given the query: '{query}'\n\nAnd the current context:\n'{context}'\n\nDoes this context provides a complete answer to the query?
                        If yes, provide the answer.
                        If no, state that the answer is in complete
                        
                        Is complete answer(Yes/No)
                        Answer(if complete):
            """
        )
        return answer_check_prompt | self.llm.with_structured_output(AnswerCheck) 
    def _retrieval_relevant_document(self, query:str):
        """
        Retrieval relavant document based on query using vectorestrore
        
        ARG:
            query str: the query to be answer
        return :
            list: a list of relevant docucument
        """
        print("Retrieval Document: .........")
        retriever = self.vectorstore.as_retriever(search_type = "similarity", search_kwargs = {'k': 5})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compressor_retriever = ContextualCompressionRetriever(base_compressor = compressor, base_retriever = retriever)
        return compressor_retriever.invoke(query)

    def _check_answer(self, query: str, context: str):
        """
        Check if the current context provides a complete answer to the query

        Args:
            Query (str): The query to be answer
            context (str): The current query
        Return:
            Tuple response:
            is_complete (bool): Whether the context provides a complete answer
            answer      (str): the answer base on context, if complete
        """
        response = self.answer_check_chain.invoke({"query": query, "context":context})
        return response.is_complete, response.answer
    def _expand_context(self, query: str, relevant_docs: list[str])->tuple[str, list[int], dict[int, str], str]:
        """
        Expands the context by traversing  the knowledge graph using Dijkstra-like approach

        This method implement a modified version of Dijkstra's algorithm to explore the knowledge graph,
        prioritizing the most relevant and strongly connect information. This algorithm work as following:

        ==================================================================================================
        
        1 Initialize:
            Initialize with nodes corresponding to the most relevant documents
            Use the priority queue to manage the traversal order, where the priority queue base one connection strenght
            Maintian dictionary is a best known distance(inverse of connection stregths) of each node
        2 Traverse:
            Always explore the node with hightest priority (strongest connection) next.
            With in node check if it find a complete answer
            Explore the node's neighbors, updating their priority if a stronger connect is found

        3 Concept Handling:
            Track visited concept to guide the exploration toward new, relevant information.
            Expand new context only if they introduce new concepts

        4 Termination:
            Stop if complete answer is found 
            Continue until the priority queue is empty (all reachable node explored)


        ==================================================================================================

        This approach ensure that:
            Priority the most relevant and strongest connection information
            Expand the concepts information
            Get the most answer by following the strongest connect informaiton in knowledge graph


        ARG:
            query (str): the query to be answer
            relevant_docs (list[Document]): List of document relevant to start the traversal
        Return:
            tuple : A tuple of a contains:
            explanded_context (str): the accumulate from traversed nodes
            treversal_path    (list[int]): The sequence of node indicate visited
            filter_content    (dict[int, str]): A mapping of node indicate to their content
            final_answer      (str): The final answer found if any    
        """

        # Initialize variable
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {} # store if best known "distance" (inverse of connection streght) to each node

        print("\nTreversing the knowledge graph")

        # Initialize with priority queue with closest node from relevant docs
        for doc in relevant_docs:
            # find the most relevant similar node in the knowledge graph for the each relevant document
            closest_nodes = self.vectorstore.similarity_search_with_score(doc.page_content, k =1)
            closest_node_content, similarity_score = closest_nodes[0]

            # Get the corresponding node in the our knowledge graph
            closest_node = next(n for n in self.knowledgegraph.graph.nodes if self.knowledgegraph.graph.nodes[n]["content"] == closest_node_content.page_content)

            # Initialize priority (inverser of similarity score for min-heap behavior)
            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        # process
        step = 0
        while priority_queue:
            # Get the node with highest priority
            current_priority, current_node = heapq.heappop(priority_queue)

            # Skip if we've already found with the hightest priority (lowest distance value)
            if current_priority > distances.get(current_node, float('inf')):
                break

            if current_node not in traversal_path:
                step+=1
                traversal_path.append(current_node)
                node_content = self.knowledgegraph.graph.nodes[current_node]["content"]
                node_concepts = self.knowledgegraph.graph.nodes[current_node]["concepts"]

                # Add node content to our acummulate context
                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # Log the current step for debugging and visualization
                print(f"step: {step}")
                print(f"current content: {node_content[:100]} ")
                print("-"*100)
                print(f"curent concept: {node_concepts}")


                # Check if we have a complete answer in the current context
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break
                
                # Process concept of the current node
                node_concepts_set = set(self.knowledgegraph._lemmatize_concept(c) for c in node_concepts)
                if node_concepts_set not in visited_concepts:
                    visited_concepts.update(node_concepts_set)


                    # Explore neighbors
                    for neighbor in self.knowledgegraph.graph.neighbors(current_node):
                        edge_data = self.knowledgegraph.graph[current_node][neighbor]
                        edge_weight = edge_data["weight"]

                    # Calculate new distance (priority) to the neighbor
                    # -Note we use 1/edge_weight because higher weights mean stronger connection
                    distance = current_priority + (1/ edge_weight)

                    # If we've found the strongest a stronger connection to the  neighbor, update its distance
                    if distance < distances.get(neighbor, float("inf")):
                        distances[neighbor] = distance
                        heapq.heappush(priority_queue,(distance, neighbor) )


                        # Process the neighbor node if it not already in our traversal path
                        if neighbor not in traversal_path:
                            step+=1
                            traversal_path.append(neighbor)
                            neighbor_content = self.knowledgegraph.graph.nodes[neighbor]["content"]
                            neighbor_concepts = self.knowledgegraph.graph.nodes[neighbor]["concepts"]

                            filtered_content[neighbor] = neighbor_content
                            expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                            # Log neighbor node  information
                            print(f"step: {step}")
                            print(f"current content: {node_content[:100]} ")
                            print("-"*100)
                            print(f"curent concept: {neighbor_concepts}")
            
                            # Check if we have a complete answer after adding the neighbor's content
                            is_complete, answer = self._check_answer(query, expanded_context)
                            if is_complete:
                                final_answer = answer
                                break

                            # Process neighbor's concept
                            neighbor_concepts_set = set(self.knowledgegraph._lemmatize_concept(c) for c in neighbor_concepts)
                            if not neighbor_concepts_set.issubset(visited_concepts):
                                visited_concepts.update(neighbor_concepts_set)
                
                # If we found the final answer, break out the main loop
                if final_answer:
                    break

        # if we haven't found a complete answer, generate one using the LLM
        if not final_answer:
            response_prompt = PromptTemplate(
                input_variables = ["query", "context"],
                template = "Base on query the following context, please answer query:\n\n Context: {context}\nQuery: {query}\nAnswer:"
            )
            response_chain = response_prompt | self.llm
            input_data ={"query" : query, "context": expanded_context}
            final_answer = response_chain.invoke(input_data)

            # input_data = {"query": query, "context": expanded_context}
            # final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer
                
    def query(self, query: str) -> tuple[str, list[int], dict[int, str]]:
        """
        Process query by retrieving relevant documents, expanding context and generate final answer

        Args:
            query str: the query to the answer
        return:
            Tuple: A tuple containing
                final answer  (str): The final answer of query
                travelsal_path (list): The traversal path of nodes in the knowledge graph
                filtered_content (dict): filtered content of nodes   
        """
        with get_openai_callback() as cb:
            print("\nProcessing query")
            relevant_docs = self._retrieval_relevant_document(query)
            expanded_context, traversal_path, filter_content, final_answer, = self._expand_context(query, relevant_docs) 

            if not final_answer:
                print("\n Generate final answer \n")
                response_prompt = PromptTemplate(
                    input_variables = ["query", "context"],
                    template = "Base on the following context, please answer the query. \n\n Context: {context}, Query: {query}\n\nAnswer:"
                )
                response_chain = response_prompt| self.llm
                input_data = {"query": query, "context": expanded_context}

                response = response_chain.invoke(input_data)
                final_answer = response
            else:
                print("\n Complete answer found during traversal")

            print(f"\nFinal Answer: {final_answer}")
            print(f"\nTotal Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")

        return final_answer, traversal_path, filter_content


"""
Define visual class 
"""
class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        """
        Visualizes the traversal path on the knowledge graph with nodes, edges, and traversal path highlighted.

        Args:
        - graph (networkx.Graph): The knowledge graph containing nodes and edges.
        - traversal_path (list of int): The list of node indices representing the traversal path.

        Returns:
        - None
        """

        traversal_graph = nx.DiGraph()

        # Add node and edge node from the original graph
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data = True):
            traversal_graph.add_edge(u, v, **data)

        # fig, ax= plt.subplot(figsize= (16,12))
        fig, ax = plt.subplots(figsize=(16, 12))
        #generate position of all nodes
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50) 

        # Draw regular edges with color based on weight
        edges = traversal_graph.edges()
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos, 
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues,
                               width=2,
                               ax=ax)

        # Draw nodes
        nx.draw_networkx_nodes(traversal_graph, pos, 
                               node_color='lightblue',
                               node_size=3000,
                               ax=ax)


        # Draw traversal path with curved arrows
        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]
            
            # Calculate control point for curve
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)
            
            # Draw curved arrow
            arrow = patches.FancyArrowPatch(start_pos, end_pos,
                                            connectionstyle=f"arc3,rad={0.3}",
                                            color='red',
                                            arrowstyle="->",
                                            mutation_scale=20,
                                            linestyle='--',
                                            linewidth=2,
                                            zorder=4)
            ax.add_patch(arrow)

        # Prepare labels for the nodes
        labels = {}
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label
        
        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''
        
        # Draw labels
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)
        
        # Highlight start and end nodes
        start_node = traversal_path[0]
        end_node = traversal_path[-1]
        
        nx.draw_networkx_nodes(traversal_graph, pos, 
                               nodelist=[start_node], 
                               node_color='lightgreen', 
                               node_size=3000,
                               ax=ax)
        
        nx.draw_networkx_nodes(traversal_graph, pos, 
                               nodelist=[end_node], 
                               node_color='lightcoral', 
                               node_size=3000,
                               ax=ax)
        
        ax.set_title("Graph Traversal Flow")
        ax.axis('off')
        
        # Add colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Edge Weight', rotation=270, labelpad=15)
        
        # Add legend
        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Regular Edge')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Traversal Path')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Start Node')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15, label='End Node')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path, filtered_content):
        """
        Prints the filtered content of visited nodes in the order of traversal.

        Args:
        - traversal_path (list of int): The list of node indices representing the traversal path.
        - filtered_content (dict of int: str): A dictionary mapping node indices to their filtered content.

        Returns:
        - None
        """
        print("\nFiltered content of visited nodes in order of traversal:")
        for i, node in enumerate(traversal_path):
            print(f"\nStep {i + 1} - Node {node}:")
            print(f"Filtered Content: {filtered_content.get(node, 'No filtered content available')[:200]}...")  # Print first 200 characters
            print("-" * 50)


class GraphRAG:
    def __init__(self):
        """
        Initialize the GraphRAG system with components for document processing, knowledge grarh with  construction, query ans visualization.

        Attributes:
            llm : An instance of LLM of generating reponses
            embedding_model: An instance of an embedding model  for document  embeddings
            document_processor: An instance of an DocumentProcessor class for processing documents
            knowledgegraph: An instance of an knowledgegraph class for building and managering the knowledge graph
            query_engine: An instance of an QueryEngine class for Handling queries (initialize as None)
            visualizer:An instance of an visualizer class for visualizing the knowledge graph traversal
        """
        self.llm = ChatOpenAI(model = "gpt-         o-mini", temperature = 0, max_tokens = 4000)
        self.embedding_model = OpenAIEmbeddings()
        self.document_processor = DocumentProcessor()
        self.knowledge_graph = Knowledge_graph()
        self.query_engine = None
        self.visualizer = Visualizer()

    def process_documents(self, documents):
        """
        Processes a list of document by splitting them into chunks, embedding them, building a knowledge graph.

        Args:
            documents (list of str): A list of document to be processed
        return:  None
        """
        splits, vectorstore = self.document_processor.process_documents(documents)
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)
        self.query_engine = QueryEngine(vectorstore, self.knowledge_graph, self.llm)

    def query(self, query: str):
        """
        Handles a query by retrieving relevant information  for the knowledge graph and visualizing the traversal path,

        args:
            query (str): the quey to be answered
        return :
            str: the response to the query
        """
        response, traversal_path, filtered_content = self.query_engine.query(query)
        if traversal_path:
            self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
        else:
            print("No traversal path to visualize.")
        
        return response


if __name__ == "__main__":

    path = "data/Understanding_Climate_Change.pdf"

    loader = PyPDFLoader(path)
    documents = loader.load()
    documents = documents[:10]
    graph_rag = GraphRAG()
    graph_rag.process_documents(documents)

    query = "what is the main cause of climate change?"
    response = graph_rag.query(query)   





"""
Processing query
Retrieval Document: .........

Treversing the knowledge graph
step: 1
current content: Chapter 2: Causes of Climate Change 
Greenhouse Gases 
The primary cause of recent climate change is 
----------------------------------------------------------------------------------------------------
curent concept: ['fossil fuels', 'greenhouse effect', 'human activities', 'energy', 'fossil fuel consumption', 'Fossil Fuels', 'carbon dioxide', 'industrial revolution', 'CH4', 'nitrous oxide', 'greenhouse gases', 'methane', 'Coal', 'climate change']
step: 2
current content: Chapter 2: Causes of Climate Change 
Greenhouse Gases 
The primary cause of recent climate change is 
----------------------------------------------------------------------------------------------------
curent concept: ['carbon pricing mechanisms', 'Local and Community Initiatives \nUrban Climate Action', 'market development', 'mandates for renewable energy use', 'efficiency standards', 'pollution controls', 'renewable energy targets', 'cap-and-trade systems', 'cost to carbon emissions', 'subsidies', 'emission reduction regulations', 'urban climate action', 'tax incentives', 'enforcement', 'carbon taxes', 'renewable energy adoption', 'carbon footprints', 'Carbon Pricing \nCarbon', 'emission caps', 'cleaner technologies', 'robust infrastructure', 'compliance', 'key sectors', 'emission reductions', 'community initiatives', 'limit emissions', 'local initiatives']

 Complete answer found during traversal

Final Answer: content='The main cause of climate change is the increase in greenhouse gases in the atmosphere, primarily due to human activities such as burning fossil fuels.' 
additional_kwargs={'refusal': None} 
response_metadata={'token_usage': {'completion_tokens': 27, 'prompt_tokens': 374, 'total_tokens': 401, 'completion_tokens_details': 
                    {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 
                    'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-4o-mini-2024-07-18', 
                    'system_fingerprint': 'fp_2b114d87de', 'id': 'chatcmpl-E9TPTP6HBlSVK3v77xTlrxzG1bvtV', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} 
                    id='lc_run--019fd186-f8ac-7292-96d9-88a5765dd190-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 374, 'output_tokens': 27, 'total_tokens': 401, 
                    'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

Total Tokens: 2754
Prompt Tokens: 2476
Completion Tokens: 278
Total Cost (USD): $0.0005382

"""