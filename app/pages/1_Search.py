import streamlit as st
import os
import fitz
from typing import Tuple, List, Dict
from backend.rag import get_answer
from backend.config import config


@st.cache_data(show_spinner=False)
def get_answer_and_cache(data_version: int, query: str, use_gpt: bool, top_k: int) -> Tuple[List[Dict], str]:
    """
    Retrieves the answer and caches it based on the data version and query parameters.

    :param data_version: Version of the data to ensure cache validity.
    :param query: The user's search query.
    :param use_gpt: Flag to determine whether to use GPT for generating the answer.
    :param top_k: Number of top results to retrieve.
    :return: A tuple containing the list of results and the LLM response.
    """
    return get_answer(query, use_gpt, top_k)


# Initialize session state variables
if "QUERY" not in st.session_state:
    st.session_state["QUERY"] = ""
if "USE_GPT" not in st.session_state:
    st.session_state["USE_GPT"] = False
if "TOP_K" not in st.session_state:
    st.session_state["TOP_K"] = 2
if "DIRS_CREATED" not in st.session_state:
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.MEDIA_FOLDER, exist_ok=True)
    st.session_state["DIRS_CREATED"] = True
if "DATA_VERSION" not in st.session_state:
    st.session_state["DATA_VERSION"] = 0

# Set page configuration
st.set_page_config(page_title="Найти информацию", page_icon="app/favicon.ico")

st.title("Поисковый бот для рекламного агентства")
st.header("Поиск информации")
query = st.text_input("Введите ваш вопрос здесь:", st.session_state["QUERY"])

col_search, col_options1, col_options2 = st.columns([2, 1, 1])

with col_search:
    search_button = st.button("Поиск")
with col_options1:
    use_gpt = st.checkbox("Использовать GPT", value=st.session_state["USE_GPT"])
with col_options2:
    top_k = st.selectbox("Количество ответов", options=[1, 2, 3, 4, 5], index=st.session_state["TOP_K"])
    st.session_state["TOP_K"] = top_k - 1

if search_button:
    if query.strip() == "":
        st.warning("Пожалуйста, введите вопрос для поиска.")
    else:
        st.session_state["QUERY"] = query
        with st.spinner("Поиск ответа..."):
            st.session_state["LIST_RESULT"] = get_answer_and_cache(
                st.session_state["DATA_VERSION"], query, use_gpt, top_k
            )
            st.rerun()  # костыль, try using forms

# Display results
if "LIST_RESULT" in st.session_state and st.session_state["LIST_RESULT"]:
    result_list, llm_response = st.session_state["LIST_RESULT"]

    if llm_response:
        st.subheader("Ответ:")
        st.write(llm_response)

    options = [f"Результат {i+1}" for i in range(len(result_list))]
    st.subheader("Вот какие файлы нашлись по вашему запросу:")
    selected_option = st.selectbox("Выбрать результат поиска", options)
    index = options.index(selected_option)
    result = result_list[index]

    file_path = os.path.join(config.MEDIA_FOLDER, result["filename"])
    col_file, col_download = st.columns([1, 7])
    with col_file:
        st.write(f"**Файл:** {result['filename']}")
    if os.path.exists(file_path):
        with col_download:
            st.download_button(
                label="Скачать файл",
                data=open(file_path, "rb").read(),
                file_name=result["filename"],
                icon=":material/arrow_downward:",
                mime="application/octet-stream"
            )

    try:
        page_number = int(result["info"])
        st.write(f"**Номер слайда/страницы:** {page_number}")
    except Exception as e:
        st.write(result['info'])

    # If it's a PDF, return the image of the page
    if result["filename"].lower().endswith(".pdf") and result.get("info"):
        if os.path.exists(file_path):
            try:
                with fitz.open(file_path) as doc:
                    if 0 <= page_number - 1 < len(doc):
                        page = doc.load_page(page_number - 1)
                        pix = page.get_pixmap()
                        image_data = pix.tobytes("png")
                        st.image(image_data, caption=f"Страница {page_number} из {result['filename']}")
                    else:
                        st.error("Неверный номер страницы.")
            except Exception as e:
                st.error(f"Ошибка при отображении страницы: {e}")
        else:
            st.error("PDF-файл не найден.")
    else:
        st.info("Превью доступно только для PDF-файлов с указанным номером страницы.")
elif "LIST_RESULT" in st.session_state:
    st.warning("Результаты не найдены.")

# Stylization
st.markdown("""
<style>
    .st-button button {
        background-color: #4CAF50;
        color: white;
    }
    .st-spinner > div > div {
        border-top-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)
