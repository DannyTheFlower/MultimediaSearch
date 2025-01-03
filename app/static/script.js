document.addEventListener('DOMContentLoaded', function () {
    const resultViewer = document.getElementById('result-viewer');
    const prevButton = document.getElementById('prev-result');
    const nextButton = document.getElementById('next-result');
    const llmResponse = document.getElementById('llm-response');

    // Массив результатов и текущий индекс
    let currentIndex = 0;
    let results = [];

    fetch('/api/results')
        .then(res => res.json())
        .then(data => {
            results = data.results;
            showResult(0);
        });

    function showResult(index) {
        currentIndex = index;
        const res = results[index];
        resultViewer.innerHTML = `
            <p><b>Файл:</b> ${res.filename}</p>
            <a href="/download/${res.filename}" class="btn btn-download">Скачать файл <i class="fa fa-download"></i></a><br>
            ${res.info ? `<p><b>Номер слайда/страницы:</b> ${res.info}</p>` : ''}
            ${res.is_pdf && res.page_number 
                ? `<div class="pdf-preview-container">
                      <iframe src="/preview/${res.filename}/${res.page_number}" class="pdf-preview"></iframe>
                   </div>` 
                : (res.is_pdf ? '<p class="info-text">Нет номера страницы.</p>' : '<p class="info-text">Превью доступно только для PDF-файлов.</p>')}
        `;

        prevButton.disabled = (index === 0);
        nextButton.disabled = (index === results.length - 1);
    }

    if (prevButton && nextButton) {
        prevButton.addEventListener('click', function() {
            if (currentIndex > 0) {
                showResult(currentIndex - 1);
            }
        });

        nextButton.addEventListener('click', function() {
            if (currentIndex < results.length - 1) {
                showResult(currentIndex + 1);
            }
        });
    }
});
