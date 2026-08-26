import os 
import sys
import warnings
import json
import numpy as np
import hashlib
from typing import List, Optional, Any

from dataclasses import dataclass, field
import networkx as nx
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

try: 
    import ollama
    OLLAMA_AVAILABLE = True
except:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available. Install with: pip install ollama")
# print(f"✅ Ollama available: {OLLAMA_AVAILABLE}")


#  "mistral", "phi3"
MODEL_NAME ="llama3.1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBEDDING_MODEL)

def call_llm(prompt: str, model: str= MODEL_NAME) ->str:
    """
    Call local LLM via Ollama
    """
    if not OLLAMA_AVAILABLE:
        return '{"entities": [], "relationships": []}'

    response = ollama.chat(
        model = model,
        messages = [{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def call_llm_json(prompt: str, model: str = MODEL_NAME)-> str:
    """
    Call llm and parse Json response
    """
    try:
        if OLLAMA_AVAILABLE:
            response = ollama.chat(
                model = model,
                messages = [{"role":"user", "content": prompt}],
                format = "json"
            )
            return json.load(response["message"]["content"])
        else:
            return {"entities":[], "relationships": []}
    except:
        return {"error": "Failed to parse JSON"}


"""
======================================================
Data Model for Graph RAG define structured classes to track: Entities, relationship, Citation
======================================================
"""
@dataclass
class Entity:
    """An entity extracted from a document with full provenance."""
    id: str
    name: str
    entity_type: str  # PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, etc.
    description: str
    source_doc: str
    source_sentence: str  # Exact sentence where entity was found
    embedding: Optional[np.ndarray] = None

@dataclass 
class Relationship:
    """A relationship between two entities with provenance."""
    source_entity: str
    target_entity: str
    relation_type: str  # WORKS_FOR, DEVELOPED, LOCATED_IN, etc.
    description: str
    source_doc: str
    source_sentence: str

@dataclass
class Citation:
    """A verifiable citation linking a claim to its source."""
    claim: str
    source_document: str
    source_sentence: str
    confidence: float
    graph_path: List[str]  # Path through the graph that led to this claim

@dataclass
class AttributedAnswer:
    """Final answer with full attribution."""
    answer: str
    citations: List[Citation]
    reasoning_trace: List[str]
    entities_used: List[str]
    relationships_traversed: List[str]
     
"""
======================================================
Knowledge Graph Manager
+ Extract entities and relationshio using LLm
+ Builds graph structured
+ Maintains source procenance for every node and edge
======================================================
"""

class KnowledgeGraphRAG:
    """
    Local Knowledge Graph with verifiable attribution,

    This class demonstrates Graph RAG that:
    1. Run entities local (privacy-first)
    2. Extract entities and relationship using LLM
    3. Enable multi-hop relationship
    4. Provides verifiable source attribution
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities : dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.documents: dict[str, str]= {}
        self.entity_embeddings: dict[str,np.ndarray] ={}
    def extract_entities_and_ralationships(self, text:str, doc_name:str)-> tuple[list[Entity], list[Relationship]]:
        """
        Use llm to extract entities and relationships.
        Each entities preserves the source sentence for attribution
        """
        extraction_prompt = """Analyze this text and extract entities and relationships.
        TEXT:
        {text}

        Return JSON with this exact structure:
        {{
        "entities": [
            {{
            "name": "Entity Name",
            "type": "PERSON|ORGANIZATION|CONCEPT|TECHNOLOGY|EVENT|LOCATION",
            "description": "Brief description",
            "source_sentence": "Exact sentence from the text where this entity appears"
            }}
        ],
        "relationships": [
            {{
            "source": "Source Entity Name",
            "target": "Target Entity Name", 
            "type": "WORKS_FOR|DEVELOPED|USES|LOCATED_IN|RELATED_TO|FOUNDED|MENTORED",
            "description": "How they are related",
            "source_sentence": "Exact sentence describing this relationship"
            }}
        ]
        }}

        IMPORTANT: 
        - Include the EXACT source_sentence from the original text
        - This is critical for verifiable attribution
        """
        result = call_llm_json(extraction_prompt)
        entities = []
        for e in result.get('entities', []):
            entity_id = self._generate_id(e['name'])
            entity = Entity(
                id=entity_id,
                name=e['name'],
                entity_type=e.get('type', 'UNKNOWN'),
                description=e.get('description', ''),
                source_doc=doc_name,
                source_sentence=e.get('source_sentence', text[:100])
            )
            entities.append(entity)
        
        relationships = []
        for r in result.get('relationships', []):
            rel = Relationship(
                source_entity=r['source'],
                target_entity=r['target'],
                relation_type=r.get('type', 'RELATED_TO'),
                description=r.get('description', ''),
                source_doc=doc_name,
                source_sentence=r.get('source_sentence', '')
            )
            relationships.append(rel)
        
        return entities, relationships
    

    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for an entity."""
        return hashlib.md5(text.lower().encode()).hexdigest()[:12]

    
    def add_document(self, text: str, doc_name: str):
        """
        Ingest a document into knowledge graph.
        Extract entities and relationships, preversing source attribution
        """
        self.documents[doc_name]= text
        print(f"Processing document: {doc_name}")
        # extract entities and relationship

        entities, relationships = self.extract_entities_and_relationships(text, doc_name)

        # add entities to graph
        for entity in entities:
            self.entities[entity.id] = entity

            # generate embedding for vector search
            embedding = embedder.encode(f'{entity.name}: {entity.description}')
            self.entity_embeddings[entity.id] =embedding
            entity.embedding = embedding
            # add node to graph with all prevenance metadata
            self.graph.add_node(
                entity.id,
                name = entity.name,
                type = entity.entity_type,
                description = entity.description,
                source_doc=entity.source_doc,
                source_sentence=entity.source_sentence
            )
        # add relationship to graph
        for rel in relationships:
            source_id = self._generate_id(rel.source_entity)
            target_id = self._generate_id(rel.target_entity)

            if source_id in self.entities and target_id in self.entitiesl:
                self.graph.add_edge(
                    source_id,
                    target_id,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    source_doc=rel.source_doc,
                    source_sentence=rel.source_sentence
                )
                self.relationships.append(rel)

        print(f"   ✅ Extracted {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships
    def vector_search(self, query, top_k = 3 ):
        """
        Find entities most similarity to the query using vector seach

        ARGs:
            query: the query string for process
            top_k: top entities similarity 
        Return:
            list[entities]
        """
        query_embedding = embedder.encode(query)
        similarities = []
        for entity_id, embedding in self.entity_embeddings.items():
            similarity = np.dot(query_embedding, embedding)/(
                np.linalg.norm(query_embedding)*np.linalg.norm(embedding)
            )
            similarities.append(entity_id, similarity)
        # sort by similarity:
        similarities.sort(key= lambda x: x[1] , reverse= True)
        return [self.entities[eid] for eid, _ in similarities[:top_k]]

    def multi_hop_traversed(self, start_entites: list[Entity], max_hops: int= 2):
        """
        Perform multi-hop graph traversed from starting entities.
        Return all reachable nodes with their paths (for attribution)
        """
        results = []
        visited = set()

        for start_entity in start_entites:
            # BFS traversal up to max_hops
            queue = [(start_entity.id, [start_entity.name], 0)]

            while queue:
                current_id, path, depth = queue.pop(0)
                if depth>max_hops and current_id in visited:
                    continue

                visited.append(current_id)

                # get node data
                if current_id in self.graph:
                    node_data = self.graph.nodes[current_id]
                    results.append({
                        "entity_id": current_id,
                        "entity_name": node_data.get("name"),
                        "description": node_data.get("description"),
                        "source_doc": node_data.get("source_doc"),
                        "source_sentence": node_data.get("source_sentence"),
                        "path":path,
                        "depth":depth
                        }
                    )
                    # explore neighbor
                    for neighbor in self.graph.neighbors(current_id):
                        edge_data = self.graph.edges[current_id, neighbor]
                        neighbor_name = self.graph.nodes[neighbor].get('name', neighbor)
                        new_path = path + [f"--[{edge_data.get('relation_type')}]-->", neighbor_name]
                        queue.append((neighbor, new_path, depth + 1))
        return results

    def hybrid_search(self, query: str,
                    vector_top_k: int = 3,
                    graph_hops : int = 2)->list[dict[str, Any]]:
        """
        Hydried search combining vector similarty and graph traversal

        1. Vector search find entry points(most relevant find entities)
        2. Graph traversed expand the context information 
        """
        # 1. vector search for entry point
        entry_points = self.vector_search(query, top_k = vector_top_k)
        print(f"Found {len(entry_points)} entry point for vector search")
        # 2. Multi-hop graph retrieval  
        traversal_results = self.multi_hop_traversed(entry_points, max_points= graph_hops)
        print(f"   🔗 Traversed to {len(traversal_results)} related entities")
        
        return traversal_results

    def  generate_attributed_answer(
            self,
            query: str,
            context: list[dict[str, Any]])->AttributedAnswer:
        """
        Generate answer with verifiable source citations

        Each claim answer to link to :
        -  Source document
        -  Source sentence
        - Graph path the led to this information
        AGRs:
            query: the query string to process
            contexts: retrival context use for generate answer

        Return:
            AttributedAnswer: (structure containing)
                answers: the generate answer
                citations: citation support for answer
                reasoning trace: the reasoning/retrieval step used
                entities used: The Entities use for the knowlege graph use to construct the answer
                relationship traversed: Graph relationship traversed
        """
        # build context string with source markers
        context_paths = []
        source_map = {}

        for i, item in enumerate(context):
            source_key = f"{i+1}"
            context_paths.append(
                f"{source_key}{item['entity_name']}:{item["description"]}"
            )
            source_map[source_key]= {
                "document":item["source_doc"],
                "sentence":item["source_sentence"],
                "path":item["path"],
            }
        context_text = "\n".join(context_paths)


        answer_prompt =f"""Base on this knowledge graph context. Answer the question.
        CRITICAL: Cite your source using [N] notation for each claim.
        CONTEXT:
        {context_text}

        QUESTION:{query}
        
        Provide a comprehensive answer with inline critical like [1], [2] for each claim
        """
        answer_text = call_llm(answer_prompt)


        # extract citation from the answer
        import re
        citation_refs = re.findall(r"(\d+)",answer_text)

        citations =[]
        for ref in set(citation_refs):
            key = f"[{ref}]"
            if key in citation_refs:
                src = source_map[key]
                citations.append(Citation(
                    claim=f"Reference {key}",
                    source_document=src['document'],
                    source_sentence=src['sentence'],
                    confidence=0.9,
                    graph_path=src['path']
                ))
        # build reasoning trace
        reasoning_trace = [
            f"🔍 Query: {query}",
            f"📌 Found {len(context)} relevant entities",
            f"🔗 Traversed graph paths",
            f"📝 Generated answer with {len(citations)} citations"
        ]
        return AttributedAnswer(
            answer= answer_text,
            citations=citations,
            reasoning_trace=reasoning_trace,
            entities_used=[c['entity_name'] for c in context],
            relationships_traversed=[]
        )



# Sample documents designed to demonstrate multi-hop reasoning
# Note: Vector RAG would need to find BOTH documents to answer questions
# about how Dr. Smith's research relates to the quantum computing project

SAMPLE_DOCUMENTS = {
    "research_team.txt": """
Dr. Sarah Chen leads the Advanced AI Research Lab at TechCorp. 
She pioneered the development of the NeuraSparse algorithm in 2022.
Dr. Chen previously studied under Professor James Miller at Stanford University.
Professor Miller is known for his foundational work in neural network optimization.
The Advanced AI Research Lab collaborates closely with the Quantum Computing Division.
Dr. Chen's team includes 15 researchers working on efficient neural architectures.
""",

    "quantum_project.txt": """
The Quantum Computing Division at TechCorp is led by Dr. Marcus Williams.
Dr. Williams implemented the NeuraSparse algorithm in their quantum error correction system.
This implementation reduced error rates by 47% compared to traditional methods.
The quantum team works on Project Aurora, funded by a $50 million grant.
Project Aurora aims to build a fault-tolerant quantum computer by 2027.
The project uses insights from Professor Miller's optimization research.
""",

    "company_overview.txt": """
TechCorp was founded in 2015 by Elena Rodriguez in San Francisco.
The company has grown to over 500 employees across three divisions.
TechCorp's three main divisions are: AI Research, Quantum Computing, and Cloud Services.
Elena Rodriguez previously worked at Google as a principal engineer.
The company's headquarters are located in the South Bay area.
TechCorp has partnerships with MIT and Stanford University for research collaboration.
"""
}


if __name__ == "__main__":
    # Initialize the Knowledge Graph RAG
    kg_rag = KnowledgeGraphRAG()

    # Ingest all documents
    for doc_name, content in SAMPLE_DOCUMENTS.items():
        kg_rag.add_document(content, doc_name)

    # print(f"\n📊 Graph Statistics:")
    # print(f"   Total Entities: {len(kg_rag.entities)}")
    # print(f"   Total Relationships: {len(kg_rag.relationships)}")
    # print(f"   Graph Nodes: {kg_rag.graph.number_of_nodes()}")
    # print(f"   Graph Edges: {kg_rag.graph.number_of_edges()}")

    """
    ==============================================
    Visual knowledge graph
    ==============================================
    """
        # Display entities
    # print("🔵 ENTITIES IN KNOWLEDGE GRAPH:\n")
    # for entity_id, entity in kg_rag.entities.items():
    #     print(f"  [{entity.entity_type}] {entity.name}")
    #     print(f"       📄 Source: {entity.source_doc}")
    #     print(f"       📝 \"{entity.source_sentence[:80]}...\"")
    #     print()

    # print("\n" + "="*60)
    # print("🔗 RELATIONSHIPS IN KNOWLEDGE GRAPH:\n")
    # for rel in kg_rag.relationships:
    #     print(f"  {rel.source_entity} --[{rel.relation_type}]--> {rel.target_entity}")
    #     print(f"       📄 Source: {rel.source_doc}")
    #     print()
        
    """
    ==============================================
    Visual knowledge graph
        - Vector search: find the relevent entry point
        - Graph traversal: Expland to connect infomation (multi-hop) 
    ==============================================
    """
    MULTI_HOP_QUERY ="How does Professor Miller's research influence the quantum computing project?"
    # Perform hybrid search
    context = kg_rag.hybrid_search(
        MULTI_HOP_QUERY, 
        vector_top_k=3, 
        graph_hops=2
    )
    print(f"\n📊 Retrieved {len(context)} relevant pieces of information:")
    for i, item in enumerate(context):
        print(f"\n  [{i+1}] {item['entity_name']}")
        print(f"      Path: {' '.join(item['path'])}")
        print(f"      Source: {item['source_doc']}")

    """
    ==============================================
    Generate answer
    ==============================================
    """
    # Generate attributed answer
    result = kg_rag.generate_attributed_answer(MULTI_HOP_QUERY, context)

    print("="*70)
    print("📝 GENERATED ANSWER WITH CITATIONS")
    print("="*70)
    print(f"\n{result.answer}")

    print("\n" + "="*70)
    print("🔍 REASONING TRACE")
    print("="*70)
    for step in result.reasoning_trace:
        print(f"  {step}")

    print("\n" + "="*70)
    print("📚 VERIFIABLE CITATIONS")
    print("="*70)
    for i, citation in enumerate(result.citations):
        print(f"\n  Citation {i+1}:")
        print(f"    📄 Document: {citation.source_document}")
        print(f"    📝 Source Sentence: \"{citation.source_sentence}\"")
        print(f"    🔗 Graph Path: {' → '.join(citation.graph_path)}")
        print(f"    ✓ Confidence: {citation.confidence:.0%}")

    """
    ==============================================
    Addition Multi_hops query
    ==============================================
    """
    # More multi-hop queries
    ADDITIONAL_QUERIES = [
        "What is the relationship between Dr. Chen and Project Aurora?",
        "How did the NeuraSparse algorithm impact quantum error correction?",
        "Who founded the company where Dr. Chen works?"
    ]

    for query in ADDITIONAL_QUERIES:
        print("\n" + "="*70)
        print(f"❓ {query}")
        print("="*70)
        
        context = kg_rag.hybrid_search(query, vector_top_k=3, graph_hops=2)
        result = kg_rag.generate_attributed_answer(query, context)
        
        print(f"\n💬 Answer:\n{result.answer[:500]}...")

"""
 Production Deployment with Neo4j

 from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Add entity with provenance
with driver.session() as session:
    session.run('''
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.source_doc = $source_doc,
            e.source_sentence = $source_sentence
    ''', id=entity_id, name=name, source_doc=doc, source_sentence=sentence)

"""