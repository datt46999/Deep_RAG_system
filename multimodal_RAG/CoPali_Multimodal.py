from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from load_dataset import convert_pdfs_to_image
from qwen_vl_utils import process_vision_info

import torch

from retriver import get_grouped_images, text_query, grouped_images, docs_retrieval_model


vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
)
vl_model.cuda().eval()

min_pixels = 224*224
max_pixels = 1024*1024
vl_model_processor = Qwen2VLProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
all_images = convert_pdfs_to_image("data")
# chat_template = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": grouped_images[0],
#             },
#             {
#                 "type": "image",
#                 "image": grouped_images[1],
#             },
#             {
#                 "type": "image",
#                 "image": grouped_images[2],
#             },
#             {
#                 "type": "text",
#                 "text": text_query
#             },
#         ],
#     }
# ]

# text = vl_model_processor.apply_chat_template(
#     chat_template, tokenize= False, add_generation_prompt = True
# )
# image_inputs, _ = process_vision_info(chat_template)
# inputs = vl_model_processor(
#     text=[text],
#     images=image_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cuda")

# generated_ids = vl_model.generate(**inputs, max_new_tokens=500)

# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = vl_model_processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text[0])

def answer_with_multimodal_rag(vl_model, 
                               docs_retrieval_model, 
                               vl_model_processor, 
                               grouped_images, 
                               text_query, 
                               top_k, 
                               max_new_tokens):
    
    result =docs_retrieval_model.search(text_query, top_k = top_k)
    grouped_images = get_grouped_images(result, all_images)
    chat_template = [
    {
      "role": "user",
      "content": [
          {"type": "image", "image": image} for image in grouped_images
            ] + [
          {"type": "text", "text": text_query}
        ],
      }
    ]

    # prepare the input
    text = vl_model_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


if __name__ == "_main__":

    output_text = answer_with_multimodal_rag(
        vl_model=vl_model,
        docs_retrieval_model=docs_retrieval_model,
        vl_model_processor=vl_model_processor,
        grouped_images=grouped_images,
        text_query="How do I assemble the Micke desk?",
        top_k=3,
        max_new_tokens=500
    )
    print(output_text[0])