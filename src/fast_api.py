from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
from eda_vectorize_function  import preprocess_text, remove_most_frequent_words, vectorize_text
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Define FastAPI app
app = FastAPI(
    title= "movie_review_sentiiment",
    version= '1.0',
    description="Provide input and see the sentiment"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("naive_bayes_model.pkl")

# Define input schema
class ReviewInput(BaseModel):
    user_review: str

@app.get("/")
def app_testing():
    return {'message':'App is succesfully started'}


# Prediction endpoint
@app.post("/predict")
def predict_sentiment(input_data: ReviewInput):
    try:
        # Preprocess the input
        processed_review = preprocess_text(input_data.user_review)
        processed_review = remove_most_frequent_words(processed_review)

        # Vectorize the review
        vectorized_review = vectorize_text(processed_review)

        # Predict
        prediction = model.predict(vectorized_review)[0]

        # Return response
        sentiment = "Positive" if prediction == 1 else "Negative"
        return {"review": input_data.user_review, "prediction": sentiment}
    except Exception as e:
        return {"error": str(e)}


if __name__== "__main__":
    print(__name__)
    uvicorn.run('fast_api:app', reload=True)