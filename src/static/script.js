const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const resultsContainer = document.getElementById('results-container');
const resultsGrid = document.getElementById('results-grid');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

async function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }

    // Update UI
    resultsContainer.classList.add('hidden');
    resultsGrid.innerHTML = '';
    loading.classList.remove('hidden');

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/search', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Search failed');

        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while searching. Make sure the backend is running and the index is built.');
    } finally {
        loading.classList.add('hidden');
    }
}

function displayResults(results) {
    resultsContainer.classList.remove('hidden');
    
    results.forEach((result, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.style.animationDelay = `${index * 0.05}s`;
        
        const similarityPercentage = (result.score * 100).toFixed(1);
        const filename = result.path.split(/[\/\\]/).pop();

        card.innerHTML = `
            <img src="${result.url}" alt="${filename}" class="result-image" loading="lazy">
            <div class="result-info">
                <span class="filename" title="${filename}">${filename}</span>
                <span class="similarity-score">${similarityPercentage}%</span>
            </div>
        `;
        
        resultsGrid.appendChild(card);
    });
}
