import streamlit as st
import time
import os
from typing import List
from backend.files_processing import upload_filedata_to_csv_file
from backend.retrieval import index_new_data, load_retrieval_resources
from backend.config import config


@st.cache_data(show_spinner=False)
def write_files(files):
    uploaded_files = []
    for file in files:
        file_path = os.path.join(config.UPLOAD_FOLDER, file.name)
        # For future: track the same namings
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        upload_filedata_to_csv_file(file_path)
        uploaded_files.append(file.name)
    return uploaded_files


if "INDEXING" not in st.session_state:
    st.session_state["INDEXING"] = False
if "DIRS_CREATED" not in st.session_state:
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.MEDIA_FOLDER, exist_ok=True)
    st.session_state["DIRS_CREATED"] = True
if "DATA_VERSION" not in st.session_state:
    st.session_state["DATA_VERSION"] = 0
processed_files: List[str] = []

st.set_page_config(page_title="Загрузить файлы", page_icon="app/favicon.ico")

st.title("Добавление новых файлов")
uploaded_files = st.file_uploader(
    "Выберите PDF или TXT файлы для загрузки",
    type=["pdf", "txt"],
    accept_multiple_files=True
)
if uploaded_files:
    if st.button("Загрузить файлы"):
        with st.spinner("Обработка файлов..."):
            processed_files.extend(write_files(uploaded_files))

if st.session_state["INDEXING"]:
    st.warning("Индексация уже запущена. Пожалуйста, подождите завершения текущей индексации.")
elif st.button("Запустить индексацию"):
    warning = st.warning("Индексация может занять продолжительное время. Пожалуйста, подождите.")
    st.session_state["INDEXING"] = True

    try:
        with st.spinner("Идёт индексация..."):
            index_new_data()

        # After indexing, move all the uploaded data to media folder
        for file in os.listdir(config.UPLOAD_FOLDER):
            src = os.path.join(config.UPLOAD_FOLDER, file)
            dst = os.path.join(config.MEDIA_FOLDER, file)

            base_name, extension = os.path.splitext(file)
            counter = 1
            while os.path.exists(dst):
                new_name = f"{base_name}_{counter}{extension}"
                dst = os.path.join(config.MEDIA_FOLDER, new_name)
                counter += 1

            os.rename(src, dst)

        # Upgrade the data
        processed_files = []
        load_retrieval_resources(st.session_state["DATA_VERSION"])
        st.success("Индексация завершена.")
    except Exception as e:
        st.error(f"Ошибка при индексации: {e}")
    finally:
        warning.empty()
        st.session_state["INDEXING"] = False

if processed_files:
    st.success("Файлы успешно загружены и обработаны:")
    for file in processed_files:
        st.write(f"- {file}")

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
