import streamlit as st
import os
from backend.retrieval import load_retrieval_resources
from backend.files_processing import load_fp_resources
from backend.config import config

st.set_page_config(page_title="Поиск-бот для Media Wise", page_icon="app/favicon.ico")

if "DIRS_CREATED" not in st.session_state:
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.MEDIA_FOLDER, exist_ok=True)
    st.session_state["DIRS_CREATED"] = True

if "INITIALIZED" not in st.session_state:
    st.session_state["INITIALIZED"] = False

if "DATA_VERSION" not in st.session_state:
    st.session_state["DATA_VERSION"] = 0

# Load embeddings, index, backup, pdf/text processors if not initialized
if not st.session_state["INITIALIZED"]:
    with st.spinner('Loading backup...'):
        load_retrieval_resources(st.session_state["DATA_VERSION"])
        load_fp_resources()
        st.session_state["INITIALIZED"] = True

st.title("Добро пожаловать в поисковый бот для Media Wise!")
st.write("Выберите страницу в боковой панели для продолжения.")
