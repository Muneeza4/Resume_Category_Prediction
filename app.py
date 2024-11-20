import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[\n\t\r]', ' ', text)  # Remove newline, tab, and carriage return characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = text.encode('ascii', 'ignore').decode('ascii')  # Handle encoding issues
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)

# Load the trained KNN model, TF-IDF vectorizer, and label encoder
try:
    with open('knn_model.pkl', 'rb') as model_file:
        knn_loaded = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer_loaded = pickle.load(vectorizer_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        le_loaded = pickle.load(le_file)
except Exception as e:
    st.write(f"Error loading models or resources: {e}")
    st.stop()  # Stop the app if the models can't be loaded

# Streamlit app
st.title('Resume Category Prediction')

# Text area for resume input
resume_text = st.text_area("Enter the resume plain text below:", max_chars=2000)

if st.button("Predict"):
    if resume_text:
        # Preprocess the input text
        processed_resume = preprocess(resume_text)

        # Transform the processed text using the loaded vectorizer
        vectorized_resume = vectorizer_loaded.transform([processed_resume])

        # Predict the category using the loaded model
        predicted_label = knn_loaded.predict(vectorized_resume)

        # Convert numerical label to categorical value
        predicted_category = le_loaded.inverse_transform(predicted_label)

        # Display the predicted category
        st.write(f"**Predicted Category:** {predicted_category[0]}")
    else:
        st.write("Please enter some resume plain text to predict.")

