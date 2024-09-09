# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.tree import DecisionTreeClassifier
# import re
# import string
# import nltk
# from nltk.corpus import stopwords

# # Download NLTK stopwords
# nltk.download('stopwords')

# # Function to load the model and vectorizer
# def load_model_and_vectorizer(model_file, vectorizer_file):
#     model = pickle.load(model_file)
#     vectorizer = pickle.load(vectorizer_file)
#     return model, vectorizer

# # Streamlit app
# st.title("Text Classification using Machine Learning")

# # Upload files
# test_text = st.text_input('give me a text to get the status of the sentense ')
# uploaded_model = st.file_uploader("Upload your model file", type="pkl")
# uploaded_vectorizer = st.file_uploader("Upload your vectorizer file", type="pkl")
# uploaded_data = st.file_uploader("Upload your dataset file (CSV format)", type="csv")

# if uploaded_model and uploaded_vectorizer and uploaded_data:
#     # Load the model and vectorizer
#     model, cv = load_model_and_vectorizer(uploaded_model, uploaded_vectorizer)

#     # Load the dataset
#     data = pd.read_csv(uploaded_data)

#     # Prepare the labels
#     if "class" in data.columns:
#         data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#         data = data[["tweet", "labels"]]

#         # Clean the text
#         stemmer = nltk.SnowballStemmer("english")
#         stopword = set(stopwords.words('english'))

#         def clean(text):
#             text = str(text).lower()
#             text = re.sub(r'\[.*?\]', '', text)
#             text = re.sub(r'https?://\S+|www\.\S+', '', text)
#             text = re.sub(r'<.*?>+', '', text)
#             text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#             text = re.sub(r'\n', '', text)
#             text = re.sub(r'\w*\d\w*', '', text)
#             text = [word for word in text.split(' ') if word not in stopword]
#             text = " ".join(text)
#             text = [stemmer.stem(word) for word in text.split(' ')]
#             text = " ".join(text)
#             return text

#         data["tweet"] = data["tweet"].apply(clean)
#         x = np.array(data["tweet"])

#         # Input box for user to enter text
#         user_input = st.text_area("Enter your text here:")

#         # Button to classify the input text
#         if st.button('Classify'):
#             if user_input:
#                 # Preprocess the input text
#                 sample_transformed = cv.transform([user_input]).toarray()
#                 # Predict the class
#                 prediction = model.predict(sample_transformed)
#                 # Define the labels
#                 labels = {0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"}
#                 result = labels.get(prediction[0], "Unknown")
#                 st.write(f"The text is classified as: {result}")
#             else:
#                 st.write("Please enter some text to classify.")
# else:
#     st.write("Please upload all required files.")






# #working

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import string

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize the NLTK components
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# Define the text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Streamlit app layout
st.title("Hate-Speech Detection with ML Model")

st.write("Enter a sample sentence to classify:")

# Input box for the sample sentence
user_input = st.text_area("Sample Sentence")

if st.button("Predict"):
    if user_input:
        # Clean the input text
        cleaned_text = clean_text(user_input)
        # Transform the text using the vectorizer
        vectorized_text = vectorizer.transform([cleaned_text])
        # Predict using the model
        prediction = model.predict(vectorized_text)
        # Display the prediction
        st.write(f"Prediction: {prediction[0]}")
    else:
        st.write("Please enter a sentence to classify.")








# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# import re
# import nltk
# import joblib
# import string
# from sklearn.metrics import accuracy_score
# import streamlit as st
# from nltk.corpus import stopwords

# # Initialize NLTK components
# nltk.download('stopwords')
# stopword = set(stopwords.words('english'))
# stemmer = nltk.SnowballStemmer("english")

# # Load dataset
# data = pd.read_csv("/home/maticz_developer_10/jegatheeswaran/twitter_data.csv")

# # Map class to labels
# data["labels"] = data["class"].map({0: "Hate Speech", 
#                                     1: "Offensive Language", 
#                                     2: "No Hate and Offensive"})

# # Prepare data
# data = data[["tweet", "labels"]]

# # Text cleaning function
# def clean(text):
#     text = str(text).lower()
#     text = re.sub('\[.*?\]', '', text)
#     text = re.sub('https?://\S+|www\.\S+', '', text)
#     text = re.sub('<.*?>+', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub('\w*\d\w*', '', text)
#     text = [word for word in text.split(' ') if word not in stopword]
#     text = " ".join(text)
#     text = [stemmer.stem(word) for word in text.split(' ')]
#     text = " ".join(text)
#     return text

# data["tweet"] = data["tweet"].apply(clean)

# x = np.array(data["tweet"])
# y = np.array(data["labels"])

# # Vectorization
# cv = CountVectorizer()
# X = cv.fit_transform(x)  # Fit the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Create and train the model
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Predict and calculate accuracy
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

# # Save the model and vectorizer
# joblib.dump(model, 'model.pkl')
# joblib.dump(cv, 'vectorizer.pkl')

# # Define the text cleaning function for Streamlit
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub('\[.*?\]', '', text)
#     text = re.sub('https?://\S+|www\.\S+', '', text)
#     text = re.sub('<.*?>+', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub('\w*\d\w*', '', text)
#     text = [word for word in text.split(' ') if word not in stopword]
#     text = " ".join(text)
#     text = [stemmer.stem(word) for word in text.split(' ')]
#     text = " ".join(text)
#     return text

# # Streamlit app layout
# st.title("Hate-Speech Detection with ML Model")

# st.write("Enter a sample sentence to classify:")

# # Input box for the sample sentence
# user_input = st.text_area("Sample Sentence", "")

# if st.button("Predict"):
#     if user_input:
#         # Clean the input text
#         cleaned_text = clean_text(user_input)
#         # Load the model and vectorizer
#         model = joblib.load('model.pkl')
#         vectorizer = joblib.load('vectorizer.pkl')
#         # Transform the text using the vectorizer
#         vectorized_text = vectorizer.transform([cleaned_text])
#         # Predict using the model
#         prediction = model.predict(vectorized_text)
#         # Display the prediction
#         st.write(f"Prediction: {prediction[0]}")
#     else:
#         st.write("Please enter a sentence to classify.")
