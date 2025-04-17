import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (e.g., IMDB dataset)
# Dataset example format: csv with 'review' and 'sentiment' columns
df = pd.read_csv('movie_reviews.csv')

# Preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_review(review):
    # Remove punctuation and convert to lowercase
    review = re.sub(r'[^\w\s]', '', review.lower())
    # Remove stop words and stem the words
    review = ' '.join(stemmer.stem(word) for word in review.split() if word not in stop_words)
    return review

df['review'] = df['review'].apply(preprocess_review)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
predictions = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
print("Model training completed.")  # Confirmation message
print("Model training completed.")  # Confirmation message
print("Saving model and vectorizer...")  # Confirmation message
print(f"Training data shape: {X_train_vec.shape}")  # Debugging output
print("Training the model...")  # Debugging output
print(f"Training data shape: {X_train_vec.shape}")  # Debugging output
print(f"Model parameters: {model.get_params()}")  # Debugging output
print(f"Vectorizer: {vectorizer}")  # Debugging output
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Vectorizer saved as 'vectorizer.pkl'")  # Confirmation message
