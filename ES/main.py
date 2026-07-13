from transformers import AutoTokenizer, AutoModelForCausalLM
from vector_search import pretty_search
from configs import model_name
def combine_query(query):
    source_information = pretty_search(query)
    return f"Query: {query}\nContinue to answer the query by using these Search Results:\n{source_information}."

def rag_query(query):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.pretrained(model_name)
    
    combine_info = combine_query(query)
    input_ids = tokenizer(combine_info, return_tensors="pt")
    response = model.generate(**input_ids, max_new_tokens = "pt")
    return tokenizer.decode(response[0], skip_special_token = True)
query = "What is the best romantic movie to watch and why?"
combined_results = combine_query(query)

print(combined_results)