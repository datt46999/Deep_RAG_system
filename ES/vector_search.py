
from elasticsearch.helpers import BulkIndexError


from configs import USE_ELASTICSEARCH_VECTORISATION, client, model_id,EMBEDDING_DIMENSIONS 
from elasticsearch import get_embedding





def vector_search(plot_query):
    if USE_ELASTICSEARCH_VECTORISATION:
        knn = {
            "field": "embedding.predicted_value",
            "k": 10,
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": model_id,
                    "model_text": plot_query,
                }
            },
            "num_candidates": 150,
        }
    else:
        question_embedding = get_embedding(plot_query)
        knn = {
            "field": "embedding",
            "query_vector": question_embedding,
            "k": 10,
            "num_candidates": 150,
        }

    response = client.search(index="movies", knn=knn, size=5)
    results = []
    for hit in response["hits"]["hits"]:
        id = hit["_id"]
        score = hit["_score"]
        title = hit["_source"]["title"]
        plot = hit["_source"]["plot"]
        fullplot = hit["_source"]["fullplot"]
        result = {
            "id": id,
            "_score": score,
            "title": title,
            "plot": plot,
            "fullplot": fullplot,
        }
        results.append(result)
    return results

def pretty_search(query):

    get_knowledge = vector_search(query)

    search_result = ""
    for result in get_knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"

    return search_result