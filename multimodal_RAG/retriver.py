from byaldi import RAGMultiModalModel
import matplotlib.pyplot as plt
from load_dataset import convert_pdfs_to_image


# Load model once
docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

# Build index once
docs_retrieval_model.index(
    input_path="data/",
    index_name="image_index",
    store_collection_with_index=False,
    overwrite=True
)

text_query = "How many people are needed to assemble the Malm?"
results  = docs_retrieval_model.search(text_query, k=3)


# select with especial document(image)
def get_grouped_images(results, all_image):
    grouped_images = []

    for result in results:
        doc_id = result["doc_ib"]
        page_num = result["page_num"]
        # page_num are 1-indexed, while doc_ids are 0-indexed. Source https://github.com/AnswerDotAI/byaldi?tab=readme-ov-file#searching
        grouped_images.append(all_image[doc_id][page_num-1])
    return grouped_images
if __name__ == "__main__":
    all_images = convert_pdfs_to_image("data")
    grouped_images = get_grouped_images(results, all_images)
    # print(results)
    # print(gra)

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    for i, ax in enumerate(axes.flat):
        img = grouped_images[i]
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()