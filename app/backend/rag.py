from backend.retrieval import find_similar_neighbors
from backend.utils import detect_language
import requests

MODEL_NAME = 'b1gjp5vama10h4due384'
IAM_TOKEN = 't1.9euelZqYnInKi5TJy8_IzpmUnsjHiu3rnpWal5CVnY2ayJXJksaJyY7NnMnl8_c-KkJG-e9iGB1D_t3z935YP0b572IYHUP-zef1656VmpeYnpeLnY_Mx5DJnJDNmcrM7_zF656VmpeYnpeLnY_Mx5DJnJDNmcrM.E2LM94R6WFFPNa6mQGFKXaWgGrliT5Y1O6Rt51Bos2x3PQnTr1g9XZRl9iaw6BSeMgwyy0_KkZIvgkPoeRBuAw'


def get_llm_answer(query, query_language, texts_for_rag):
    global MODEL_NAME, IAM_TOKEN

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


def get_answer(query, use_gpt=True, topk_for_rag=3):
    rows_for_rag = find_similar_neighbors(query, topk_for_rag)
    info = []
    texts_for_rag = []
    for row in rows_for_rag:
        d = {'filename': row.filename, 'info': row.n_slide if row.n_slide is not None else row.text}  # hopefully it works
        info.append(d)
        texts_for_rag.append(row.text)
    answer = get_llm_answer(query, detect_language(query), texts_for_rag) if use_gpt else None
    return info, answer
