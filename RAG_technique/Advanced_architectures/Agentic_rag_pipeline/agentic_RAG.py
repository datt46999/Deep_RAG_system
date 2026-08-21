"""
Learning Outcomes :
Traditional take the queries as-is, often leading to poor retrievel for amibiguous, context lacking or complex query. Agentic RAG intelligently preprocesses queries to bridge this gap. 
In the query path the primary agentic step is query reformulation, comprising-turn, query expansion, query decomposition.
This wuery reformulation step is critical to obtaining the most RAG result, and is one component of a system engineered to generate most accuracy response.

In the query reformulation, context is added or queries are restructured from the orginal input:
     for multi-turn, adding iterative dialogue context; 
     for query expansion, adding additional context to help short query return optim result;
     for query decomposition, taking complex multi-faceted quries that require reasoning across several unrelative document and breaking them down into several subqueries that help obtain the most relevant retrieval.

"""

import os
import json
import requests
from pathlib import Path
from typing import List, Optional, Dict
from IPython.display import display, JSON
import pandas as pd
from contextual import ContextualAI
import ast
from IPython.display import display, Markdown
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/Agentic_RAG.ipynb