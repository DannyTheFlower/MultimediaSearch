import streamlit as st
import os
import fitz

from backend.rag import get_answer


@st.cache_data(show_spinner=False)
def get_answer_and_cache(data_version, query, use_gpt, top_k):
    return get_answer(query, use_gpt, top_k)


if "QUERY" not in st.session_state:
    st.session_state["QUERY"] = ""
if "USE_GPT" not in st.session_state:
    st.session_state["USE_GPT"] = False
if "TOP_K" not in st.session_state:
    st.session_state["TOP_K"] = 2
if "UPLOAD_FOLDER" not in st.session_state:
    st.session_state["UPLOAD_FOLDER"] = "uploaded_files"
    os.makedirs(st.session_state["UPLOAD_FOLDER"], exist_ok=True)
if "MEDIA_FOLDER" not in st.session_state:
    st.session_state["MEDIA_FOLDER"] = "media"
    os.makedirs(st.session_state["MEDIA_FOLDER"], exist_ok=True)
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

            st.session_state["LIST_RESULT"] = get_answer_and_cache(st.session_state["DATA_VERSION"], query, use_gpt, top_k)

if "LIST_RESULT" in st.session_state and st.session_state["LIST_RESULT"]:
    result_list, llm_response = st.session_state["LIST_RESULT"]

    if llm_response:
        st.subheader("Ответ:")
        st.write(llm_response)
    options = [f"Результат {i+1}" for i in range(len(result_list))]
    st.subheader("Вот, какие файлы нашлись по вашему запросу:")
    selected_option = st.selectbox("Выбрать результат поиска", options)
    index = options.index(selected_option)
    result = result_list[index]

    file_path = os.path.join(st.session_state["MEDIA_FOLDER"], result["filename"])
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
