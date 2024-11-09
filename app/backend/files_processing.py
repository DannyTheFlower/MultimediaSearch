from utils import get_file_extension
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from easyocr import Reader
import os
import time
import torch
import csv 
import numpy as np
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
import fitz
import pandas as pd

nltk.download('punkt', quiet=True)


CSV_FILE_PATH = 'indexed_data.csv'
OCR = None 
CHART2TEXT = None 


class OCRModel:
    def __init__(self):
        self.reader = Reader(['en', 'ru'], gpu=torch.cuda.is_available())

    def extract_text(self, image: Image.Image) -> str:
        image_np = np.array(image)
        results = self.reader.readtext(image_np)
        text = ' '.join([result[1] for result in results])
        return text


class Chart2Text:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma",
                                                                       torch_dtype=torch.float16).to(self.device)
        self.processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

    def extract_text(self, image: Image.Image) -> str:
        input_text = "program of thought: describe the chart in great detail"
        inputs = self.processor(text=input_text, images=image, return_tensors="pt").to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        generate_ids = self.model.generate(**inputs, num_beams=4, max_new_tokens=512)
        output_text = self.processor.batch_decode(
            generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
    

def init_all():
    global OCR, CHART2TEXT
    OCR = OCRModel()
    CHART2TEXT = Chart2Text()


def get_text_pieces_from_txt_file(
        filepath: str,
        chunk_size: int = 250,
        overlap: int = 10
):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = word_tokenize(text)
    total_tokens = len(tokens)
    chunks = []
    start = 0
    while start < total_tokens:
        chunk_tokens = tokens[start:min(start + chunk_size, total_tokens)]
        chunks.append(" ".join(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def get_text_pieces_from_pdf_file(
        filepath: str,
        too_few_chars: int = 50
):
    global OCR, CHART2TEXT

    doc = fitz.open(filepath)
    text_pieces = []
    for page_number, page in enumerate(doc):
        page_text = page.get_text()
        if len(page_text.strip()) < too_few_chars and page_number not in [0, len(doc) - 1]:
            # Use OCR for text extraction if too few characters
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_text = OCR.extract_text(img)
            if len(page_text.strip()) < too_few_chars:
                # Use CHART2TEXT if still too few characters
                page_text = CHART2TEXT.extract_text(img)
        text_pieces.append(page_text)
    return text_pieces


def write_data_to_csv(data):
    global CSV_FILE_PATH
    while True:
        try:
            file_exists = os.path.isfile(CSV_FILE_PATH)
            with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['filename', 'n_slide', 'text'])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'filename': data['filename'],
                    'n_slide': data['n_slide'],
                    'text': data['text'],
                })
            break
        except IOError:
            time.sleep(1)
            continue
            

def upload_filedata_to_csv_file(
        filepath: str,
        too_few_chars: int = 50,
        chunk_size: int = 250,
        overlap: int = 10,
):
    filename = os.path.basename(filepath)
    extension = get_file_extension(filename)
    if extension == '.txt':
        text_pieces = get_text_pieces_from_txt_file(filepath, chunk_size, overlap)
    elif extension == '.pdf':
        text_pieces = get_text_pieces_from_pdf_file(filepath, too_few_chars)
    else:
        raise ValueError('Unsupported file extension')
    for idx, text_piece in enumerate(text_pieces):
            csv_row = {'filename': filename, 'n_slide': None, 'text': text_piece}
            write_data_to_csv(csv_row)
            