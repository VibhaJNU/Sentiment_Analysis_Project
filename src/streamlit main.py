import streamlit as st
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Movie Sentiment Analyzer 🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# App heading
st.title("🎭 Movie Review Sentiment Analyzer")
st.markdown("Enter your review and discover if it's **Positive** or **Negative**!")

# Input box
user_review = st.text_area("✍️ Your Movie Review:", height=150, placeholder="Type your movie review here...")

# Button to predict
if st.button("🔍 Analyze Sentiment") and user_review.strip() != "":
    with st.spinner("Analyzing sentiment..."):
        try:
            # API URL
            url = "http://127.0.0.1:8000/predict"  # Make sure FastAPI is running at this port 
            # Payload
            payload = json.dumps({"user_review": user_review})
            headers = {"Content-Type": "application/json"}

            # POST request
            response = requests.post(url, headers=headers, data=payload)

            if response.status_code == 200:
                result = response.json()
                sentiment = result.get("prediction", "Unknown")
                review = result.get("review", "")

                if sentiment == "Positive":
                    st.success("🎉 Positive Sentiment Detected!")
                    st.markdown(f"**Your Review:** _{review}_")
                elif sentiment == "Negative":
                    st.error("😞 Negative Sentiment Detected.")
                    st.markdown(f"**Your Review:** _{review}_")
                else:
                    st.warning("Could not detect sentiment.")
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"🚨 Exception: {str(e)}")
else:
    st.caption("⬆️ Enter a review and click the button to analyze.")

# Optional: Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 12px;'>Built with ❤️ using FastAPI + Streamlit</div>",
    unsafe_allow_html=True
)


# #Pre process the data
# processed_review = preprocess_text(user_review)
# processed_review = remove_most_frequent_words(processed_review) 

# # Vectorize the processed review
# vectorize_review = vectorize_text(processed_review)

# # Load the pre-trained model
# model = joblib.load("naive_bayes_model.pkl")

# # Predict the sentiment of the review
# prediction = model.predict(vectorize_review)

# # Display the prediction
# if prediction[0] == 1:
#     st.write("The sentiment of your review is: Positive 😊")
# else:
#     st.write("The sentiment of your review is: Negative 😞")
