from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import English
import numpy as np
from responses import friendly_responses  # Import the friendly responses from responses.py

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize spaCy's English tokenizer
nlp = English()
tokenizer = nlp.tokenizer

# Sample training data for demonstration
training_texts = ["Hello, how can I help you?", "Goodbye", "Can you assist me with my account?", "Thank you!"]
training_labels = ["greeting", "farewell", "account_help", "gratitude"]

# Custom text preprocessing function
def preprocess_text_spacy(text: str):
    doc = tokenizer(text.lower())
    return [token.text for token in doc if token.is_alpha]

# Initialize vectorizer and fit on sample data
vectorizer = CountVectorizer(tokenizer=preprocess_text_spacy)
X_train = vectorizer.fit_transform(training_texts)

# Initialize and train a basic decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, training_labels)

# Define API endpoint for predicting response
@app.route('/ai/message', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Message content is required"}), 400

    # Preprocess and predict
    processed_input = preprocess_text_spacy(user_input)
    input_features = vectorizer.transform([" ".join(processed_input)])
    predicted_response = model.predict(input_features)[0]

    # Get a friendly response based on the prediction
    response = {
        "response": np.random.choice(friendly_responses.get(predicted_response, ["I'm not sure how to respond to that. Can you clarify?"]))
    }

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(port=5000)
