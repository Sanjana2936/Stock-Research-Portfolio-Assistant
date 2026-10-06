document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('research-form');
    const input = document.getElementById('query-input');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    const submitBtn = document.getElementById('submit-btn');
    const resultsContainer = document.getElementById('results-container');
    const markdownContent = document.getElementById('markdown-content');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');

    // Handle suggestion chips clicking
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            input.value = chip.textContent;
            input.focus();
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = input.value.trim();
        if (!query) return;

        // UI Loading state
        submitBtn.disabled = true;
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        
        try {
            // Call the FastAPI backend
            const response = await fetch('http://127.0.0.1:8000/api/research', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Render markdown to HTML using marked.js
            markdownContent.innerHTML = marked.parse(data.report);
            
            // Show results
            resultsContainer.classList.remove('hidden');
            
            // Smooth scroll to results
            setTimeout(() => {
                resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);

        } catch (error) {
            console.error('Error:', error);
            markdownContent.innerHTML = `<p style="color: #ef4444;">Sorry, an error occurred while generating the report. Please make sure the backend server is running and API keys are set.</p>`;
            resultsContainer.classList.remove('hidden');
        } finally {
            // Restore UI state
            submitBtn.disabled = false;
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    });
});
