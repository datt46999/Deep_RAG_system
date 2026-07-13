from elasticsearch  import helpers
from elasticsearch.helpers import BulkIndexError

from sentence_transformers import SentenceTransformer

from configs import USE_ELASTICSEARCH_VECTORISATION, EMBEDDING_MODEL_ID, EMBEDDING_DIMENSIONS, client
from data_loader import load_datasets
index_name = "movies"

if not USE_ELASTICSEARCH_VECTORISATION:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

def get_embedding(text: str) -> list[float]:
    if USE_ELASTICSEARCH_VECTORISATION:
        raise Exception(
            f"Disabled when USE_ELASTICSEARCH_VECTORISATION is [{USE_ELASTICSEARCH_VECTORISATION}]"
        )
    else:
        if not text.strip():
            print("Attempted to get embedding for empty text.")
            return []

        embedding = embedding_model.encode(text)
        return embedding.tolist()
    

def add_fullplot_embedding(x):
    if USE_ELASTICSEARCH_VECTORISATION:
        raise Exception(
            f"Disabled when USE_ELASTICSEARCH_VECTORISATION is [{USE_ELASTICSEARCH_VECTORISATION}]"
        )
    else:
        full_plots = x["fullplot"]
        return {"embedding": [get_embedding(full_plot) for full_plot in full_plots]}
    

def index_vector_mapping():
    
    index_mapping = {
    "properties": {
        "fullplot": {"type": "text"},
        "plot": {"type": "text"},
        "title": {"type": "text"},
        }
    }
    # define index mapping
    if USE_ELASTICSEARCH_VECTORISATION:
        index_mapping["properties"]["embedding"] = {
            "properties": {
                "is_truncated": {"type": "boolean"},
                "model_id": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "predicted_value": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMENSIONS,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    else:
        index_mapping["properties"]["embedding"] = {
            "type": "dense_vector",
            "dims": EMBEDDING_DIMENSIONS,
            "index": "true",
            "similarity": "cosine",
        }


    return index_mapping


def idx_setting( model_id):
    should_delete_index = True

    # check if we want to delete index before creating the index
    if should_delete_index:
        if client.indices.exists(index=index_name):
            print("Deleting existing %s" % index_name)
            client.indices.delete(index=index_name, ignore=[400, 404])

    print("Creating index %s" % index_name)
    if USE_ELASTICSEARCH_VECTORISATION:
        pipeline_id = "vectorize_fullplots"

        client.ingest.put_pipeline(
            id=pipeline_id,
            processors=[
                {
                    "inference": {
                        "model_id": model_id,
                        "target_field": "embedding",
                        "field_map": {"fullplot": "text_field"},
                    }
                }
            ],
        )

        index_settings = {
            "index": {
                "default_pipeline": pipeline_id,
            }
        }
    else:
        index_settings = {}
    
    return index_settings



def batch_to_bulk_actions(batch):
    for record in batch:
        action ={
            "_index": "movies",
            "_source":{
                "title": record["title"],
                "fullplot": record["fullplot"],
                "plot": record["plot"],
            }
        }
        if not USE_ELASTICSEARCH_VECTORISATION:
            action["_source"]["embedding"] = record["embedding"]
        yield action


def bulk_index(ds):
    start = 0
    end = len(ds)
    batch_size = 100
    if USE_ELASTICSEARCH_VECTORISATION:
        # If using auto-embedding, bulk requests can take a lot longer,
        # so pass a longer request_timeout here (defaults to 10s), otherwise
        # we could get Connection timeouts
        batch_client = client.options(request_timeout=600)
    else:
        batch_client = client
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        print(f"batch: start [{batch_start}], end [{batch_end}]")
        batch = ds.select(range(batch_start, batch_end))
        actions = batch_to_bulk_actions(batch)
        helpers.bulk(batch_client, actions)


if __name__ =="__main__":
    index_mapping = index_vector_mapping()
    index_setting = idx_setting()
    client.options(ignore_status=[400, 404]).indices.create(
    index=index_name, mappings=index_mapping, settings=index_setting
    )
    dataset = load_datasets()
    try:
        bulk_index(dataset["train"])
    except BulkIndexError as e:
        print(f"{e.errors}")