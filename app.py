from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        review = data.get('review')
        
        if not review:
            return jsonify({'error': 'Review is required'}), 400
        
        # Preprocess and vectorize the review
        X = vectorizer.transform([review])
        
        # Make prediction
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        confidence = np.max(probability) * 100
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
app = Flask(__name__)
CORS(app)

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    review = data['review']
    
    # Preprocess and vectorize the review
    X = vectorizer.transform([review])
    
    # Make prediction
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    confidence = np.max(probability) * 100
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)