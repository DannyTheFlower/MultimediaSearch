import streamlit as st
import time
import os
import fitz

if "QUERY" not in st.session_state:
    st.session_state["QUERY"] = ""
if "UPLOAD_FOLDER" not in st.session_state:
    st.session_state["UPLOAD_FOLDER"] = "uploaded_files"
    os.makedirs(st.session_state["UPLOAD_FOLDER"], exist_ok=True)
if "MEDIA_FOLDER" not in st.session_state:
    st.session_state["MEDIA_FOLDER"] = "media"
    os.makedirs(st.session_state["MEDIA_FOLDER"], exist_ok=True)

# Set page configuration
st.set_page_config(page_title="Найти информацию", page_icon="app/favicon.ico")

# Поиск информации
st.title("Поисковый чат-бот для рекламного агентства")
st.header("Поиск информации")
query = st.text_input("Введите ваш вопрос здесь:", st.session_state["QUERY"])

# Кнопка для отправки запроса
if st.button("Поиск"):
    if query.strip() == "":
        st.warning("Пожалуйста, введите вопрос для поиска.")
    else:
        st.session_state["QUERY"] = query
        with st.spinner("Поиск ответа..."):
            @st.cache_data(show_spinner=False)
            def search_answer(query):
                time.sleep(2)
                return [
                    {
                        "answer": "Сократилось более чем на 50%",
                        "filename": "../media/1.pdf",
                        "slide_number": 34
                    },
                    {
                        "answer": "Количество телепрограмм увеличилось на 20%",
                        "filename": "data_analysis.txt",
                        "slide_number": None
                    },
                    {
                        "answer": "Рейтинги телепрограмм снизились в два раза",
                        "filename": "report2.pdf",
                        "slide_number": 12
                    }
                ]

            st.session_state["LIST_RESULT"] = search_answer(query)

if "LIST_RESULT" in st.session_state and st.session_state["LIST_RESULT"]:
    result_list = st.session_state["LIST_RESULT"]

    options = [f"Результат {i+1}" for i in range(len(result_list))]
    selected_option = st.selectbox("Выберите результат для просмотра:", options)
    index = options.index(selected_option)
    result = result_list[index]

    st.subheader("Ответ:")
    st.write(result["answer"])
    st.write(f"**Файл:** {result['filename']}")
    if result['slide_number']:
        st.write(f"**Номер слайда/страницы:** {result['slide_number']}")

    # Если это PDF-файл, отображаем нужную страницу
    if result["filename"].lower().endswith(".pdf") and result.get("slide_number"):
        pdf_path = os.path.join(st.session_state['UPLOAD_FOLDER'], result["filename"])
        page_number = result["slide_number"]

        if os.path.exists(pdf_path):
            try:
                with fitz.open(pdf_path) as doc:
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
            st.error("PDF файл не найден.")
    else:
        st.info("Превью доступно только для PDF-файлов с указанным номером страницы.")
elif "LIST_RESULT" in st.session_state:
    st.warning("Результаты не найдены.")

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
