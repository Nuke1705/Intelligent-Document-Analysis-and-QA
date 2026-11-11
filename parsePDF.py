
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unstructured_pytesseract
from pathlib import Path

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Verify that the executable exists
assert os.path.isfile(tesseract_path), (
    f"Tesseract not found at: {tesseract_path}"
)

# Tell Unstructured's OCR wrapper where Tesseract is
unstructured_pytesseract.pytesseract.tesseract_cmd = (
    tesseract_path
)
# Keep your existing partition_pdf() code below

from unstructured.partition.pdf import partition_pdf

# Get the images from the CompositeElement objects
def get_images_base64(chunks, file, images, captions):
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for i, el in enumerate(chunk_els):
                if "Image" in str(type(el)):
                    images.append({"data": el.metadata.image_base64, "source": file})
                    
                    # Prevent another potential IndexError if an Image is the absolute last element
                    if i + 1 < len(chunk_els) and "FigureCaption" in str(type(chunk_els[i+1])):
                        captions.append({"data": chunk_els[i+1].text, "source": file})
                        
                        # Use -2 to check the previously added caption
                        if len(captions) > 1 and captions[-2] == "None":
                            captions[-2] = chunk_els[i+1].text
                    else:
                        captions.append({"data": "None", "source": file})
   


# Get the formula from the CompositeElement objects
def get_formulas_base64(chunks, file, formulas):
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Formula" in str(type(el)):
                    formulas.append({"data": el.metadata.image_base64, "source": file})

def extract(file, output_path, tables, texts, images, captions, formulas):
    chunks = partition_pdf(
        filename=(output_path + file),
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image", "Formula"],   # Add 'Table' to list to extract image of tables
        image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,

        # extract_images_in_pdf=True,          # deprecated
    ) 

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append({"data": chunk, "source": file})

        if "CompositeElement" in str(type((chunk))):
            texts.append({"data": chunk, "source": file})

    get_images_base64(chunks, file, images, captions)
    get_formulas_base64(chunks, file, formulas)

    return

def parsepdf(outputpath):
    output_path = Path(outputpath)
    pdf_files = list(p.name for p in output_path.glob("*.pdf"))

    # separate tables from texts
    tables = []
    texts = []
    images = []
    captions = []
    formulas = []   

    for file in pdf_files:
        extract(file, outputpath, tables, texts, images, captions, formulas)
        print(file + " extracted")

    return tables, texts, images, formulas, captions