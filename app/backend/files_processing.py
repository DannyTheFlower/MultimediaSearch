from utils import detect_language, get_file_extension
from retrieval import init_all as retrieval_init_all
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from easyocr import Reader
import os
import shutil
import time
import torch
import csv 
import numpy as np
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
import fitz

nltk.download('punkt', quiet=True)


CSV_FILE_PATH = 'indexed_data.csv'
OCR = None 
CHART2TEXT = None 


class OCRModel:
    """
    OCR class for extracting text from images using EasyOCR.

    Methods:
        extract_text(image): Extracts text from a given image.
    """

    def __init__(self):
        self.reader = easyocr.Reader(['en', 'ru'], gpu=torch.cuda.is_available())

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text from a given image using EasyOCR.

        Parameters:
            image (Image.Image): The image to extract text from.

        Returns:
            str: The extracted text.
        """
        image_np = np.array(image)
        results = self.reader.readtext(image_np)
        text = ' '.join([result[1] for result in results])
        return text


class Chart2Text:
    """
    CHART2TEXT class for extracting text descriptions from chart images.

    Methods:
        extract_text(image): Extracts descriptive text from a chart image.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma",
                                                                       torch_dtype=torch.float16).to(self.device)
        self.processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts descriptive text from a chart image using the Chart2Text model.

        Parameters:
            image (Image.Image): The chart image to process.

        Returns:
            str: The extracted descriptive text.
        """
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


def split_text_into_chunks(text: str, chunk_size: int = 250, overlap: int = 10):
    tokens = word_tokenize(text)
    total_tokens = len(tokens)
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def get_text_pieces_from_txt_file(
        filepath: str,
        chunk_size: int = 250,
        overlap: int = 15
):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    return chunks


def get_text_pieces_from_pdf_file(
        filepath: str,
        too_few_chars: int = 100
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
    """
    Writes data to the global CSV file. Handles concurrent access by using a lock.

    Parameters:
        data (CSVRow): The data to write to the CSV file.
    """
    while True:
        try:
            file_exists = os.path.isfile(CSV_FILE_PATH)
            with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'n_slide', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'filename': data['filename'],
                    'n_slide': data['n_slide'],
                    'text': data['text'],
                })
            break
        except IOError:
            # Wait and retry if the file is being accessed by another process
            time.sleep(1)
            continue
            

def upload_filedata_to_csv_file(
        filepath: str,
        too_few_chars: int = 50,
        chunk_size: int = 250,
        overlap: int = 15,
):
    filename = os.path.basename(filepath)
    extension = get_file_extension(filename)

    if extension == '.txt':
        text_pieces = get_text_pieces_from_txt_file(filepath, chunk_size, overlap)
        for idx, text_piece in enumerate(text_pieces):
            csv_row = {'filename': filename, 'n_slide': None, 'text': text_piece}
            write_data_to_csv(csv_row)
        
    elif extension == '.pdf':
        text_pieces = get_text_pieces_from_pdf_file(filepath, too_few_chars)
        for idx, text_piece in enumerate(text_pieces):
            csv_row = {'filename': filename, 'n_slide': idx+1, 'text': text_piece}
            write_data_to_csv(csv_row)
    else:
        raise ValueError('Unsupported file extension')


def index_filepaths(filepaths):
    for filepath in filepaths:
        upload_filedata_to_csv_file(filepath)
    
    shutil.rmtree('index')  # create new index
    retrieval_init_all(save_index_path='index.faiss')
    