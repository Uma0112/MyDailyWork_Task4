import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from nltk.corpus import stopwords
import nltk

# Set page configuration (must be the first command)
st.set_page_config(page_title="SMS Spam Classifier", layout="wide")

# Download NLTK resources
nltk.download('stopwords')

# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load data
data = load_data()
data['message'] = data['message'].apply(preprocess_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Print model performance to console
print("Model Performance")
print("Model Accuracy:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

# Define and train the best model
best_model = VotingClassifier(estimators=[
    ('nb', MultinomialNB()),
    ('lr', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True)),
    ('rf', RandomForestClassifier())
], voting='soft')

best_model.fit(X_train_tfidf, y_train)

# Streamlit UI
st.title("📩 SMS Spam Classifier")
st.subheader("Classify your SMS messages as Spam or Ham")

# User input for custom message
user_input = st.text_area("Enter a message to classify:")

# Button to classify the message
if st.button("Classify Message"):
    if user_input:
        # Preprocess user input
        user_input_tfidf = tfidf.transform([preprocess_text(user_input)])
        
        # Predict using the best model
        prediction = best_model.predict(user_input_tfidf)[0]
        result = "Spam" if prediction == 1 else "Ham"
        
        # Display the result
        st.success(f"✅ This message is classified as: **{result}**")
    else:
        st.warning("⚠️ Please enter a message to classify.")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, best_model.predict(X_test_tfidf))
print("Confusion Matrix:")
print(conf_matrix)

# Word Cloud for Spam Messages
st.subheader("Word Cloud for Spam Messages")
spam_words = " ".join(data[data['label'] == 1]['message'])
wordcloud = WordCloud(width=600, height=400).generate(spam_words)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Footer
st.markdown("---")
st.write("### About This App")
st.write("This application uses machine learning to classify SMS messages as spam or ham. "
         "It utilizes various models and visualizations to provide insights into the classification process.")
