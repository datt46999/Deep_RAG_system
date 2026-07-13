import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()
CLOUD_ID = os.getenv()
ELASTIC_API_KEY =os.getenv()
client = Elasticsearch(cloud_id=CLOUD_ID, api_key=ELASTIC_API_KEY)



USE_ELASTICSEARCH_VECTORISATION = False

EMBEDDING_MODEL_ID = "thenlper/gte-small"
# https://huggingface.co/thenlper/gte-small's page shows the dimensions of the model
# If you use the `gte-base` or `gte-large` embedding models, the numDimension
# value in the vector search index must be set to 768 and 1024, respectively.
EMBEDDING_DIMENSIONS = 384

model_id = EMBEDDING_MODEL_ID.replace("/", "__")

model_name ="google/gemma-2b-it"