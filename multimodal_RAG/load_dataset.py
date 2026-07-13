import requests
import os
from pdf2image import convert_from_path

# pdfs = {
#     "MALM": "https://www.ikea.com/us/en/assembly_instructions/malm-4-drawer-chest-white__AA-2398381-2-100.pdf",
#     "BILLY": "https://www.ikea.com/us/en/assembly_instructions/billy-bookcase-white__AA-1844854-6-2.pdf",
#     "BOAXEL": "https://www.ikea.com/us/en/assembly_instructions/boaxel-wall-upright-white__AA-2341341-2-100.pdf",
#     "ADILS": "https://www.ikea.com/us/en/assembly_instructions/adils-leg-white__AA-844478-6-2.pdf",
#     "MICKE": "https://www.ikea.com/us/en/assembly_instructions/micke-desk-white__AA-476626-10-100.pdf"
# }

# output_dir = "data"

# for name, url in pdfs.items():
#     response = requests.get(url)
#     pdf_path = os.path.join(output_dir, f"{name}.pdf")

#     with open(pdf_path, "wb") as f:
#         f.write(response.content)
#     print(f"Downloaded {name} to {pdf_path}")

# print("Downloaded files:", os.listdir(output_dir))

def convert_pdfs_to_image(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    all_images = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        all_images[doc_id] = images

    return all_images


if __name__ =="__main__":
    all_images = convert_pdfs_to_image("data")
    import matplotlib.pyplot as plt 
    fig, axes = plt.subplots(1,8, figsize =(15,10))
    for i, ax in enumerate(axes.flat):
        img = all_images[0][i]
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()