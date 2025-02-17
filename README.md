# Media Wise Searcher
### Краткое описание решения: 
Был разработан прототип поисковой системы по мультимодальным данным (текст, изображения) с привлечением русскоязычной языковой модели для агрегации результатов поисковой выдачи. Поисковая система оптимизирована под запросы на чтение, но также поддерживает запись (загрузка pdf и txt файлов).
 
### Сценарий вопрос-ответ:
1. При поступлении запроса система обращается к хранилищу данных для поиска релевантных документов. 
2. С помощью YandexGPT формируется ответ на основе найденных документов и запроса пользователя. 
3. Возвращается ответ вместе с указанием источников.
### Сценарий добавления файлов:
1. В зависимости от формата файла происходит обработка:
* Файлы pdf обрабатывается инструментами PyMuPDF, EasyOCR и VLM
2. Извлеченный из PDF и TXT файлов текст разбивается на чанки и индексируется с помощью BERT-like эмбеддера
3. Создается индекс в векторной базе данных Qdrant для быстрого поиска

### Технические особенности: 
* OCR для распознавания текста на изображении
* Блок retrieval с Qdrant для быстрого поиска
* Использование мультиязычного индексатора
* Модель для получения текстового описания слайдов
* Возможность пользователя выбрать количество возвращаемых документов, а также указать, хочет ли он получать ответ от YandexGPT
* Решение поддерживает английский и русский языки, ответ предоставляется на том же языке, на котором написан запрос

### Запуск

1. Склонируйте репозиторий и перейдите в корневую папку проекта
2. Установите необходимые библиотеки из `requirements.txt`
3. Запустите базу данных Qdrant (хост и порт можно использовать по умолчанию), восстановите коллекции из снапшотов из директории `app/backend/backup`
4. Создайте файл `.env` в корне проекта и запишите `LLM_MODEL_NAME` и `LLM_IAM_TOKEN` для работы с YandexGPT в следующем формате:

   ```
   LLM_MODEL_NAME=your-model-name
   LLM_IAM_TOKEN=your-iam-token
   ```

   При необходимости переопределите и другие настройки (например, если вы запускали Qdrant не на дефолтных настройках)
   
5. В терминале запустите команду `fastapi run main.py`
