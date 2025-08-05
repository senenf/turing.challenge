import fitz  # PyMuPDF
import os
import sys
import re
import base64
import openai

from PIL import Image
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    filename = os.path.basename(pdf_path)
    for i in range(len(doc)):
        for img in doc.get_page_images(i):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                from io import BytesIO
                test_image = Image.open(BytesIO(image_bytes))
                test_image.verify()

                # Save extracted image to file
                image_name = f"{filename}_page{i+1}.{image_ext}"
                save_path = os.path.join(pdf_path.replace(".pdf", "_extracted_images"), image_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)

                images.append({
                    "image_path": save_path,
                    "page": i + 1,
                    "source_pdf": filename
                })

            except Exception as e:
                print(f"⚠️ Skipping image on page {i+1} due to error: {e}")
                continue
    return images



# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv("../src/utils/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load helper functions from your pipeline
sys.path.append(os.path.abspath("../src"))
from pipeline.indexer import index_documents

# Generate caption using GPT-4V
def caption_image_with_gpt4v(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes images accurately."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one or two sentences."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# Extract page number from filename
def extract_page_number(filename: str) -> int:
    match = re.search(r'page(\d+)', filename)
    return int(match.group(1)) if match else -1

# Process images in a folder (or single image)
def process_images(image_metadata_list: List[dict]) -> List[Document]:
    documents = []

    for img_meta in image_metadata_list:
        image_path = img_meta["image_path"]
        page = img_meta["page"]
        source_pdf = img_meta["source_pdf"]

        filename = os.path.basename(image_path)
        print(f"📄 Processing: {filename} (page {page})")

        try:
            caption = caption_image_with_gpt4v(image_path)
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

        if not caption.strip():
            print(f"⚠️ Empty caption for {filename}. Skipping.")
            continue

        doc = Document(
            page_content=caption,
            metadata={
                "page_number": page,
                "image_file": filename,
                "source": source_pdf,   
                "type": "image",
                "path": image_path
            }
        )
        documents.append(doc)

    return documents


import os
def index_image_documents(image_folder_path: str, collection_name: str = "knowledge_base"):
    if not os.path.isdir(image_folder_path):
        print(f"⚠️ Path is not a folder: {image_folder_path}")
        return

    image_files = [
        os.path.join(image_folder_path, f)
        for f in os.listdir(image_folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))
    ]

    if not image_files:
        print("⚠️ No image files found in folder.")
        return

    # 🔁 Rebuild metadata expected by process_images
    source_pdf = os.path.basename(image_folder_path).replace("_extracted_images", ".pdf")
    image_metadata_list = []
    for image_path in image_files:
        filename = os.path.basename(image_path)
        page = extract_page_number(filename)
        image_metadata_list.append({
            "image_path": image_path,
            "page": page,
            "source_pdf": source_pdf
        })

    docs = process_images(image_metadata_list)
    valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

    if not valid_docs:
        print("⚠️ No valid documents to index.")
        return

    index_documents(valid_docs, persist_directory="../src/db", collection_name=collection_name)
    print(f"✅ Indexed {len(valid_docs)} image documents into '{collection_name}' collection.")
