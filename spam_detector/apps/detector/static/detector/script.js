document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
 
    const message = document.getElementById('message').value;
    const response = await fetch('/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
        },
        body: `message=${encodeURIComponent(message)}`
    });
 
    if (response.ok) {
        const result = await response.json();
        document.getElementById('prediction-result').innerText = `Prediction: ${result.prediction}`;
    } else {
        document.getElementById('prediction-result').innerText = 'Error: Failed to predict';
    }
}); 