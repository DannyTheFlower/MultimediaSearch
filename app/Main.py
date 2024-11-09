import streamlit as st
import os

from backend.retrieval import init_all

st.set_page_config(page_title="Поиск-бот для Media Wise", page_icon="app/favicon.ico")

with st.spinner('Loading data...'):
    init_all(index_path='app/backend/index.faiss')

if "UPLOAD_FOLDER" not in st.session_state:
    st.session_state["UPLOAD_FOLDER"] = "uploaded_files"
    os.makedirs(st.session_state["UPLOAD_FOLDER"], exist_ok=True)
if "MEDIA_FOLDER" not in st.session_state:
    st.session_state["MEDIA_FOLDER"] = "media"
    os.makedirs(st.session_state["MEDIA_FOLDER"], exist_ok=True)

st.title("Добро пожаловать в поисковый бот для Media Wise!")
st.write("Выберите страницу в боковой панели для продолжения.")
