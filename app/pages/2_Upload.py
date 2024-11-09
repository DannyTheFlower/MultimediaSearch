import streamlit as st
import time
import os

from backend.files_processing import index_filepaths

if "INDEXING" not in st.session_state:
    st.session_state["INDEXING"] = False
if "UPLOAD_FOLDER" not in st.session_state:
    st.session_state["UPLOAD_FOLDER"] = "uploaded_files"
    os.makedirs(st.session_state["UPLOAD_FOLDER"], exist_ok=True)
if "MEDIA_FOLDER" not in st.session_state:
    st.session_state["MEDIA_FOLDER"] = "media"
    os.makedirs(st.session_state["MEDIA_FOLDER"], exist_ok=True)
processed_files = []

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
            @st.cache_data(show_spinner=False)
            def upload_files(files):
                processed_files = []
                for file in files:
                    file_path = os.path.join(st.session_state['UPLOAD_FOLDER'], file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    processed_files.append(file.name)
                return processed_files
            processed_files.extend(upload_files(uploaded_files))

if st.session_state["INDEXING"]:
    st.warning("Индексация уже запущена. Пожалуйста, подождите завершения текущей индексации.")
elif st.button("Запустить индексацию"):
    warning = st.warning("Индексация может занять продолжительное время. Пожалуйста, подождите.")
    st.session_state["INDEXING"] = True

    try:
        with st.spinner("Идёт индексация..."):
            index_filepaths(processed_files)

        for file in os.listdir(st.session_state["UPLOAD_FOLDER"]):
            src = os.path.join(st.session_state["UPLOAD_FOLDER"], file)
            dst = os.path.join(st.session_state["MEDIA_FOLDER"], file)

            base_name, extension = os.path.splitext(file)
            counter = 1
            while os.path.exists(dst):
                new_name = f"{base_name}_{counter}{extension}"
                dst = os.path.join(st.session_state["MEDIA_FOLDER"], new_name)
                counter += 1

            os.rename(src, dst)

        processed_files = []
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

# Стилизация приложения
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
