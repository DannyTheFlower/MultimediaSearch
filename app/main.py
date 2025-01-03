from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
from typing import List
from backend.retrieval import load_retrieval_resources, index_new_data
from backend.files_processing import load_fp_resources, prepare_filedata_for_qdrant
from backend.config import config
from backend.rag import get_answer
import fitz

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

STATE = {
    "DIRS_CREATED": False,
    "INITIALIZED": False,
    "DATA_VERSION": 0,
    "INDEXING": False,
    "QUERY": "",
    "USE_GPT": False,
    "TOP_K": 3,
    "LIST_RESULT": None
}


def init_dirs():
    if not STATE["DIRS_CREATED"]:
        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(config.MEDIA_FOLDER, exist_ok=True)
        STATE["DIRS_CREATED"] = True


def init_resources():
    if not STATE["INITIALIZED"]:
        load_retrieval_resources(STATE["DATA_VERSION"])
        load_fp_resources()
        STATE["INITIALIZED"] = True


@app.on_event("startup")
def startup_event():
    init_dirs()
    init_resources()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request,
                query: str = "",
                use_gpt: bool = False):
    results = STATE["LIST_RESULT"]
    top_k = STATE["TOP_K"]
    return templates.TemplateResponse("search.html", {
        "request": request,
        "query": query,
        "use_gpt": use_gpt,
        "top_k": top_k,
        "results": results,
    })


@app.post("/search", response_class=HTMLResponse)
def search_action(request: Request,
                  query: str = Form(""),
                  use_gpt: str = Form(None),
                  top_k: int = Form(2)):
    use_gpt_bool = True if use_gpt == "on" else False
    query = query.strip()

    if query == "":
        return RedirectResponse(url="/search?warn=empty", status_code=303)
    else:
        STATE["QUERY"] = query
        STATE["USE_GPT"] = use_gpt_bool
        STATE["TOP_K"] = top_k

        result_list, llm_response = get_answer(query, use_gpt_bool, top_k)
        STATE["LIST_RESULT"] = (result_list, llm_response)

        return RedirectResponse(url="/search", status_code=303)


@app.get("/download/{filename}", response_class=FileResponse)
def download_file(filename: str):
    file_path = os.path.join(config.MEDIA_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename=filename)
    else:
        return HTMLResponse("Файл не найден", status_code=404)


@app.get("/preview/{filename}/{page_number}", response_class=HTMLResponse)
def preview_page(request: Request, filename: str, page_number: int):
    file_path = os.path.join(config.MEDIA_FOLDER, filename)
    if os.path.exists(file_path) and filename.lower().endswith(".pdf"):
        try:
            with fitz.open(file_path) as doc:
                if 0 <= page_number - 1 < len(doc):
                    page = doc.load_page(page_number - 1)
                    pix = page.get_pixmap()
                    image_data = pix.tobytes("png")
                    # Возвращаем в base64 или через StaticFiles (можно временно сохранять)
                    # Для простоты вернём напрямую как <img src="data:image/png;base64,...">
                    import base64
                    encoded = base64.b64encode(image_data).decode()
                    img_tag = f'<img src="data:image/png;base64,{encoded}" alt="Страница {page_number} из {filename}"/>'
                    return HTMLResponse(img_tag)
                else:
                    return HTMLResponse("Неверный номер страницы.", status_code=400)
        except Exception as e:
            return HTMLResponse(f"Ошибка при отображении страницы: {e}", status_code=500)
    else:
        return HTMLResponse("Превью доступно только для PDF-файлов с указанным номером страницы или файл не найден.", status_code=400)


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "indexing": STATE["INDEXING"]})


@app.post("/upload", response_class=HTMLResponse)
async def upload_files(request: Request, files: List[UploadFile] = File([])):
    uploaded_files = []
    for file in files:
        file_path = os.path.join(config.UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        uploaded_files.append(file.filename)

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "uploaded_files": uploaded_files,
        "indexing": STATE["INDEXING"]
    })


@app.post("/index", response_class=HTMLResponse)
def start_indexing(request: Request):
    if STATE["INDEXING"]:
        # Уже идёт индексация
        return templates.TemplateResponse("upload.html", {
            "request": request,
            "indexing": True,
            "warning": "Индексация уже запущена."
        })
    else:
        STATE["INDEXING"] = True
        # Индексация
        try:
            data_rows = []
            files = os.listdir(config.UPLOAD_FOLDER)
            for file in files:
                file_path = os.path.join(config.UPLOAD_FOLDER, file)
                data_rows.extend(prepare_filedata_for_qdrant(file_path))

            index_new_data(data_rows)

            # Переносим файлы в MEDIA_FOLDER
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

            # Перезагружаем ресурсы
            load_retrieval_resources(STATE["DATA_VERSION"])
            STATE["INDEXING"] = False

            return templates.TemplateResponse("upload.html", {
                "request": request,
                "indexing": False,
                "success": "Индексация завершена."
            })
        except Exception as e:
            STATE["INDEXING"] = False
            return templates.TemplateResponse("upload.html", {
                "request": request,
                "indexing": False,
                "error": f"Ошибка при индексации: {e}"
            })


@app.get("/api/results")
def get_results():
    results_data = []
    if STATE.get("LIST_RESULT"):
        result_list, llm_response = STATE["LIST_RESULT"]
        for res in result_list:
            filename = res["filename"]
            info = res["info"]
            is_pdf = filename.lower().endswith(".pdf")
            page_number = None
            # Попытка привести info к номеру страницы
            try:
                page_number = int(info)
            except:
                pass

            results_data.append({
                "filename": filename,
                "info": info,
                "is_pdf": is_pdf,
                "page_number": page_number
            })

    return JSONResponse({"results": results_data})
