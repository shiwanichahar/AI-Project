async function analyzeSentiment() {
    const review = document.getElementById('review').value;
    const resultDiv = document.getElementById('result');
    
    if (!review) {
        resultDiv.textContent = "Please enter a review first!";
        return;
    }

    try {
        resultDiv.textContent = "Analyzing...";
        
        const response = await fetch('http://localhost:5000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Received response:", data); // Check response in console
        
        resultDiv.textContent = `Sentiment: ${data.sentiment} (${data.confidence.toFixed(2)}%)`;
        resultDiv.className = data.sentiment.toLowerCase();
        
    } catch (error) {
        console.error('Error:', error);
        resultDiv.textContent = `Error: ${error.message}`;
        resultDiv.className = 'error';
    }
}