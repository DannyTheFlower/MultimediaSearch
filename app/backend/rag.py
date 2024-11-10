from backend.retrieval import find_similar_neighbors
from backend.utils import detect_language
from backend.config import config
import requests
import numpy as np
from typing import List
import streamlit as st



def get_llm_answer(query: str, query_language: str, texts_for_rag: List[str]) -> str:
    """
    Generates an answer using a Language Model API.

    :param query: The user's query.
    :param query_language: Detected language of the query.
    :param texts_for_rag: Context texts retrieved for RAG.
    :return: Generated answer.
    """
    MODEL_NAME = st.secrets.llm["MODEL_NAME"]
    IAM_TOKEN = st.secrets.llm["IAM_TOKEN"]

    if query_language == 'ru':
        prompt = f'''Тебе даны вопрос {query} и тексты {texts_for_rag}. 
        Верни суммаризацию ответа из текстов. 
        Не выдумай ничего сам, бери информацию только из текста. 
        Ответь на русском языке'''
    elif query_language == 'en':
        prompt = f'''You are given a question {query} and texts {texts_for_rag}. 
        Return the summation of the response from the texts. 
        Do not invent anything yourself, take information only from the text. 
        Answer in english'''

    url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'

    headers = {
        'Authorization': f'Bearer {IAM_TOKEN}',
        'Content-Type': 'application/json'
    }

    data = {
        "modelUri": f"gpt://{MODEL_NAME}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": 2000
        },
        "messages": [
            {
                "role": "system",
                "text": "Ты — умный ассистент."
            },
            {
                "role": "user",
                "text": prompt
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        answer = response.json()['result']['alternatives'][0]['message']['text']
    else:
        answer = "Something went wrong when generating answer, we are sorry :("
    return answer


def get_answer(query: str, use_gpt: bool = True, topk_for_rag: int = 3):
    """
    Retrieves the answer for a query, optionally using GPT.

    :param query: The user's query.
    :param use_gpt: Flag to determine whether to use GPT.
    :param topk_for_rag: Number of top results to consider.
    :return: Tuple of information and the answer.
    """
    rows_for_rag = find_similar_neighbors(query, topk_for_rag)
    info = []
    texts_for_rag = []
    for row in rows_for_rag:
        d = {'filename': row.filename, 'info': row.n_slide if not np.isnan(row.n_slide) else row.text}
        info.append(d)
        texts_for_rag.append(row.text)
    answer = get_llm_answer(query, detect_language(query), texts_for_rag) if use_gpt else None
    return info, answer
