import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')



# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.encode('ascii', 'ignore').decode('ascii')
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load the trained KNN model, TF-IDF vectorizer, and label encoder
with open('knn_model.pkl', 'rb') as model_file:
    knn_loaded = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer_loaded = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as le_file:
    le_loaded = pickle.load(le_file)

# Streamlit app
st.title('Resume Category Prediction')

# Text area for resume input
resume_text = st.text_area("Enter the resume text below:")

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
        st.write("Please enter some resume text to predict.")
