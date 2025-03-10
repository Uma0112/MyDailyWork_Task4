# MyDailyWork_Task4

# 📩 SMS Spam Classifier

## 🚀 Project Overview
This project is a **Spam SMS Detection** system that classifies messages as either **Spam** or **Ham (legitimate)** using various machine learning models. The application is built using **Streamlit** for a user-friendly web interface and employs techniques such as **TF-IDF vectorization** and classifiers like **Naive Bayes, Logistic Regression, SVM, and Random Forest** to enhance spam detection accuracy.

## 🏗️ Features
- **Preprocessing**: Cleans and tokenizes text by removing punctuation, stopwords, and numbers.
- **Machine Learning Models**: Trains multiple classifiers including:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Voting Classifier (Best Model)
- **Streamlit UI**: Provides an interactive interface for users to input SMS messages and get predictions.
- **Word Cloud Visualization**: Displays common words in spam messages for better insight.

## 📂 Dataset
The model is trained on the **Spam SMS Dataset** (`spam.csv`) from the UCI Machine Learning Repository. The dataset consists of:
- `v1`: Label (ham/spam)
- `v2`: Message text

## ⚙️ Installation
### Prerequisites
Ensure you have **Python 3.7+** installed along with the required libraries.

### Steps to Set Up the Project
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/spam-classifier.git
   cd spam-classifier
   ```
2. **Create a virtual environment (optional but recommended):**
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Streamlit App:**
   ```sh
   streamlit run app.py
   ```

## 🧑‍💻 Model Training and Evaluation
### **Preprocessing Steps**
- Remove punctuation and numbers.
- Convert text to lowercase.
- Remove stopwords using NLTK.
- Apply **TF-IDF Vectorization**.

### **Model Training**
The dataset is split into **80% training** and **20% testing**. The following models are trained:
- **Naive Bayes**
- **Logistic Regression**
- **SVM**
- **Random Forest**
- **Voting Classifier** (combining multiple models for best performance)

### **Evaluation Metrics**
Each model is evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**

## 🎨 Streamlit UI Overview
- **Enter a message** in the text box to classify.
- Click the **"Classify Message"** button.
- See the classification result (**Spam or Ham**).
- View **Word Cloud** for spam messages.

## 📊 Example Output
```sh
Model Accuracy:
Naive Bayes: 0.97
Logistic Regression: 0.98
SVM: 0.96
Random Forest: 0.95
Voting Classifier: 0.98
```

## 🤖 Technologies Used
- **Python** (pandas, numpy, re, string, nltk, sklearn, matplotlib, seaborn)
- **Machine Learning** (TF-IDF, Naive Bayes, Logistic Regression, SVM, Random Forest, Voting Classifier)
- **Streamlit** (For UI development)
- **WordCloud** (For visualization)

## 📌 Future Improvements
- Integrate **deep learning models** for improved accuracy.
- Deploy the app as a **web service (AWS/GCP)**.
- Implement **real-time SMS classification** using an API.

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit PRs.

## 📝 License
This project is **open-source** under the **MIT License**.

---
🚀 **Try the SMS Spam Classifier and protect yourself from spam messages!**

