**Overview**
This directory contains source code for preprocessing text, training a Naive Bayes model, vectorization, and serving the sentiment analysis via FastAPI.

🔧 **Structure**
src/
1. eda_vectorize_function.py
2. streamlit_main.py
3. fast_api.py
4. vectorizer.pkl   # saved vectorizer file
5. naive_bayes_model.pkl  # trained model file
6. requirements.txt

📝 **File Descriptions**
Filename	Purpose
eda.py	Contains text preprocessing functions: preprocess_text, remove_most_frequent_words, vectorize_text.
fast_api.py	Implements FastAPI app (/predict) for sentiment inference using the saved model and vectorizer.
vectorizer.pkl	Serialized CountVectorizer used for transforming text.
naive_bayes_model.pkl	Trained Naive Bayes model used to predict sentiment.
requirements.txt	Lists dependencies to run preprocessing, model training, and the API.

🚀 **Getting Started**
1. Install Dependencies
Activate your virtual environment and run:
pip install -r src/requirements.txt

Save naive_bayes_model.pkl and vectorizer.pkl in src/

2. **Start the API Server**
Navigate to your project root, then run:

uvicorn src.fast_api:app --reload --port 8080
Access the API at http://127.0.0.1:8080

3 **Start the streamlit application**

streamlit run \streamlit_main.py

Interactive docs available at http://127.0.0.1:8080/docs

⚙️ **Usage**
✅ Predicting Sentiment via HTTP POST request:
Endpoint: POST /predict

Payload:
{
  "user_review": "I loved the movie!"
}
Response:
{
  "review": "I loved the movie!",
  "prediction": "Positive"
}
🧩 **How It Works**
Text Preprocessing

Lowercasing, URL removal, punctuation cleanup

Stopword removal and lemmatization

Filtering out the most frequent words

Vectorization

Uses loaded CountVectorizer from vectorizer.pkl

Prediction

**Applies the Naive Bayes model to generate sentiment prediction**
📚 Future Enhancements
Add support for predict_proba to return probabilities

Upgrade models: integrate TfidfVectorizer, SVM or Neural Networks

Dockerize the FastAPI + model artifacts for scalable deployment
