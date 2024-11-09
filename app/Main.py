import streamlit as st
import os

from backend.retrieval import init_all as retrieval_init_all
from backend.files_processing import init_all as fp_init_all

st.set_page_config(page_title="Поиск-бот для Media Wise", page_icon="app/favicon.ico")

if "UPLOAD_FOLDER" not in st.session_state:
    st.session_state["UPLOAD_FOLDER"] = "uploaded_files"
    os.makedirs(st.session_state["UPLOAD_FOLDER"], exist_ok=True)
if "MEDIA_FOLDER" not in st.session_state:
    st.session_state["MEDIA_FOLDER"] = "media"
    os.makedirs(st.session_state["MEDIA_FOLDER"], exist_ok=True)
if "INITIALIZED" not in st.session_state:
    st.session_state["INITIALIZED"] = False

if not st.session_state["INITIALIZED"]:
    with st.spinner('Loading data...'):
        retrieval_init_all()
        fp_init_all()
        st.session_state["INITIALIZED"] = True

st.title("Добро пожаловать в поисковый бот для Media Wise!")
st.write("Выберите страницу в боковой панели для продолжения.")
