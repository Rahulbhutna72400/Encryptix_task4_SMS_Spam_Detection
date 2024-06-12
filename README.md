# SMS Spam Detection

This project aims to classify SMS messages as spam or legitimate using various machine learning techniques. The dataset used in this project is sourced from the [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset).
![image](https://github.com/Rahulbhutna72400/SMS_Spam_Detection/assets/114943689/526f0bc1-0560-4ba9-9626-9de2931f8022)
![image](https://github.com/Rahulbhutna72400/SMS_Spam_Detection/assets/114943689/bb4a47a7-7c18-425c-96ab-f0db1ef0bcc0)

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/sms-spam-detection.git
   cd sms-spam-detection

2. Install the required packages:
  pip install -r requirements.txt

Sure, here is the complete README file with detailed descriptions for each section under the Table of Contents. This version integrates all necessary instructions and explanations in a single code block:

markdown
Copy code
# SMS Spam Detection

This project aims to classify SMS messages as spam or legitimate using various machine learning techniques. The dataset used in this project is sourced from the [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/sms-spam-detection.git
   cd sms-spam-detection
Install the required packages:
sh
Copy code
pip install -r requirements.txt

##Data
The dataset consists of SMS messages labeled as either ham (legitimate) or spam. The data is stored in a CSV file named spam.csv. You can download the dataset from Kaggle SMS Spam Collection Dataset and place it in your project's directory.

##Preprocessing
The preprocessing steps include:

Cleaning the text data by removing punctuation and stop words.
Stemming the words to their root forms.
Tokenizing the messages.

##Feature Extraction
Two feature extraction methods are used:

TF-IDF: Term Frequency-Inverse Document Frequency is used to convert text data into numerical vectors.
Word2Vec: Word embeddings are used to capture the semantic meaning of the words in the messages.

##Model Training and Evaluation
Several classifiers are trained and evaluated on the dataset:

Logistic Regression
Random Forest
Gradient Boosting
XGBoost
Naive Bayes
The models are trained using both TF-IDF and Word2Vec features, and their performance is evaluated using accuracy, confusion matrix, and classification report.


Here's the comprehensive README file:

markdown
Copy code
# SMS Spam Detection

This project aims to classify SMS messages as spam or legitimate using various machine learning techniques. The dataset used in this project is sourced from the [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/sms-spam-detection.git
   cd sms-spam-detection
Install the required packages:
sh
Copy code
pip install -r requirements.txt
Data
The dataset consists of SMS messages labeled as either ham (legitimate) or spam. The data is stored in a CSV file named spam.csv. You can download the dataset from Kaggle SMS Spam Collection Dataset and place it in your project's directory.

Preprocessing
The preprocessing steps include:

Cleaning the text data by removing punctuation and stop words.
Stemming the words to their root forms.
Tokenizing the messages.
Example code for preprocessing:

python
Copy code
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('path/to/spam.csv', encoding='ISO-8859-1')

# Drop unnecessary columns and rename columns
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.rename(columns={"v1": "result", "v2": "messages"}, inplace=True)

# Preprocessing
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))
corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z0-9]', ' ', df.iloc[i]["messages"])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in stop]
    corpus.append(review)
Feature Extraction
Two feature extraction methods are used:

TF-IDF: Term Frequency-Inverse Document Frequency is used to convert text data into numerical vectors.
Word2Vec: Word embeddings are used to capture the semantic meaning of the words in the messages.
Example code for feature extraction:

python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform([' '.join(tokens) for tokens in corpus]).toarray()

# Word2Vec
model = Word2Vec(corpus, window=5, min_count=2, vector_size=20)
def get_vector(tokens, model):
    valid_words = [model.wv[word] for word in tokens if word in model.wv]
    if len(valid_words) > 0:
        return np.mean(valid_words, axis=0)
    else:
        return np.zeros(model.vector_size)
X_word2vec = np.array([get_vector(tokens, model) for tokens in corpus])
X_word2vec = np.nan_to_num(X_word2vec)
Model Training and Evaluation
Several classifiers are trained and evaluated on the dataset:

Logistic Regression
Random Forest
Gradient Boosting
XGBoost
Naive Bayes
The models are trained using both TF-IDF and Word2Vec features, and their performance is evaluated using accuracy, confusion matrix, and classification report.

Example code for training and evaluation:

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split data
y = df['result'].map({'ham': 0, 'spam': 1}).values
X_train_w2v, X_test_w2v, y_train, y_test = train_test_split(X_word2vec, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers_w2v = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate classifiers with Word2Vec features
for name, clf in classifiers_w2v.items():
    clf.fit(X_train_w2v, y_train)
    y_pred_w2v = clf.predict(X_test_w2v)
    print(f'--- {name} (Word2Vec) ---')
    print(f'Accuracy: {accuracy_score(y_test, y_pred_w2v)}')
    print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred_w2v)}')
    print(f'Classification Report:\n {classification_report(y_test, y_pred_w2v)}')
    print('\n')
    
##Visualization
Various visualizations are included to better understand the dataset:

Distribution of message lengths
Count of spam vs. ham messages
Box plot of message lengths for spam vs. ham
Word clouds for spam and ham messages

##Results
The performance of the classifiers is evaluated and compared. The best performing model is selected based on its accuracy and other evaluation metrics.

##License
This project is licensed under the MIT License. See the LICENSE file for details.
