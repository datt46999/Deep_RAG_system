from datasets import load_dataset


def load_datasets():
    dataset = load_dataset("MongoDB/embedded_movies")
    dataset =dataset.filter(lambda x: x["fullplot"] is not None)
    if "plot_embedding" in sum(dataset.column_names.values(), []):
        # Remove the plot_embedding from each data point in the dataset as we are going to create new embeddings with an open source embedding model from Hugging Face
        dataset = dataset.remove_columns("plot_embedding")

    return dataset