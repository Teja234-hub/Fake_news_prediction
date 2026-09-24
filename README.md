# 📰 Fake News Prediction using NLP and Machine Learning

An NLP-based machine learning project that classifies news-related text as **Real News** or **Fake News** using text preprocessing, TF-IDF feature extraction, and Logistic Regression.

## 📌 Project Overview

Fake news and misinformation can spread rapidly through digital platforms. This project explores how **Natural Language Processing (NLP)** and **supervised machine learning** can be used to classify news-related content into two categories:

- `0` → Real News
- `1` → Fake News

The project implements an end-to-end machine learning pipeline covering **data loading, data inspection, missing-value handling, text preprocessing, feature engineering, TF-IDF vectorization, model training, evaluation, and prediction**.

---

## 🎯 Objectives

- Perform text preprocessing on news-related data.
- Handle missing values in the dataset.
- Create a textual feature from available news information.
- Convert textual information into numerical features.
- Apply TF-IDF vectorization for feature extraction.
- Train a Logistic Regression classifier.
- Evaluate model performance on training and testing data.
- Demonstrate prediction on a news sample.

---

## 📊 Dataset

The dataset contains **20,800 news records** with the following columns:

| Column | Description |
|---|---|
| `id` | Unique identifier for each news record |
| `title` | Title of the news article |
| `author` | Author of the article |
| `text` | News article text |
| `label` | Target label: `0 = Real`, `1 = Fake` |

### Current Feature Representation

In the current implementation, the model's textual input is created by combining:

```text
author + title
```

The `text` column is present in the dataset but is **not currently included in the model's feature representation**.

---

## 🔄 Machine Learning Pipeline

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Missing Value Handling
   ↓
Feature Creation
(Author + Title)
   ↓
Text Preprocessing
   ↓
Remove Non-Alphabetic Characters
   ↓
Convert to Lowercase
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Porter Stemming
   ↓
TF-IDF Vectorization
   ↓
Train/Test Split
   ↓
Logistic Regression
   ↓
Model Training
   ↓
Prediction
   ↓
Accuracy Evaluation
```

---

## 🧹 Text Preprocessing

The project applies the following preprocessing steps:

1. Check the dataset for missing values.
2. Replace missing values with empty strings.
3. Combine `author` and `title` into a single `content` feature.
4. Remove non-alphabetic characters using regular expressions.
5. Convert text to lowercase.
6. Split the text into individual words.
7. Remove English stopwords using NLTK.
8. Apply Porter stemming.
9. Join the processed tokens back into a single text string.

### Example

```text
Original:
"The government has announced several new policies."

After preprocessing:
"government announc sever new polici"
```

---

## 🔤 Natural Language Processing

Natural Language Processing is used to convert unstructured textual information into a representation that can be processed by a machine learning algorithm.

The preprocessing pipeline consists of:

```text
Raw Text
   ↓
Cleaning
   ↓
Lowercasing
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming
   ↓
Cleaned Text
```

---

## 🛑 Stopword Removal

Common English words such as:

```text
the
is
and
of
to
in
```

are removed using the English stopword list provided by **NLTK**.

The purpose is to reduce words that may provide relatively little discriminative information for the classification task.

---

## 🌱 Stemming

The project uses **Porter Stemmer** from NLTK.

Stemming attempts to reduce different forms of words to a common stem.

For example:

```text
playing
played
plays
```

may be reduced toward:

```text
play
```

This helps reduce variation in word forms before feature extraction.

---

## 🔢 TF-IDF Vectorization

Machine learning algorithms cannot directly process raw text. Therefore, the cleaned text is converted into numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF assigns importance to words based on:

- How frequently a word appears within a document.
- How frequently or rarely the word appears across the collection of documents.

The basic idea is:

```text
Text
 ↓
TF-IDF
 ↓
Numerical Feature Matrix
 ↓
Machine Learning Model
```

The project uses:

```python
TfidfVectorizer()
```

from Scikit-learn.

---

## 🤖 Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as the binary classification model.

The target classes are:

```text
0 → Real News
1 → Fake News
```

The model receives the TF-IDF feature representation as input and learns patterns associated with the two classes.

### Why Logistic Regression?

Logistic Regression is suitable for this project because:

- The problem is a binary classification task.
- It works well with high-dimensional numerical features.
- It is commonly used as a baseline for text classification.
- It can work effectively with sparse TF-IDF feature matrices.

---

## 🧪 Train/Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The project uses a stratified split:

```python
train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

### Why Stratification?

Stratification helps maintain approximately the same distribution of target classes in both the training and testing datasets.

### Why `random_state`?

A fixed random state makes the data split reproducible.

---

## 📈 Model Performance

The current notebook records the following accuracy values:

| Dataset | Accuracy |
|---|---:|
| Training Data | 98.66% |
| Testing Data | 97.91% |

These values represent the results recorded in the current notebook implementation and dataset split.

> **Note:** These accuracy values should not be interpreted as proof that the model can reliably determine whether real-world news is factually true or false.

---

## 🔍 Prediction

After training, the Logistic Regression model can classify a processed news sample.

The prediction is interpreted as:

```text
Prediction = 0
→ Real News

Prediction = 1
→ Fake News
```

The current notebook demonstrates prediction using a sample from the test dataset.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **NLTK**
- **Scikit-learn**
- **TF-IDF**
- **Logistic Regression**
- **Jupyter Notebook**
- **Google Colab**
- **Git**
- **GitHub**

---

## 📚 Libraries Used

### Pandas

Used for:

- Loading the CSV dataset.
- Data inspection.
- Missing-value handling.
- Data manipulation.

### NumPy

Used as part of the numerical/data-science environment.

### NLTK

Used for:

- English stopword removal.
- Porter stemming.

### Scikit-learn

Used for:

- Train/test splitting.
- TF-IDF vectorization.
- Logistic Regression.
- Model evaluation.

---

## 🏗️ End-to-End System Flow

```text
                NEWS DATASET
                     │
                     ▼
              Load CSV File
                     │
                     ▼
             Inspect Dataset
                     │
                     ▼
          Check Missing Values
                     │
                     ▼
          Handle Missing Values
                     │
                     ▼
       Create Content Feature
          Author + Title
                     │
                     ▼
          Text Preprocessing
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 Remove Non-Alphabetic      Lowercase
 Characters
          │                     │
          └──────────┬──────────┘
                     ▼
                Tokenization
                     │
                     ▼
             Stopword Removal
                     │
                     ▼
              Porter Stemming
                     │
                     ▼
              Cleaned Text
                     │
                     ▼
             TF-IDF Vectorizer
                     │
                     ▼
          Numerical Feature Matrix
                     │
                     ▼
             Train/Test Split
                80% / 20%
                     │
                     ▼
           Logistic Regression
                     │
                     ▼
                Model Training
                     │
                     ▼
                 Prediction
                     │
                     ▼
             Real / Fake Class
                     │
                     ▼
              Model Evaluation
```

---

## 📁 Project Structure

```text
Fake_news_prediction/
│
├── Fake_News_Prediction.ipynb
│
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Teja234-hub/Fake_news_prediction.git
```

### 2. Navigate to the Project

```bash
cd Fake_news_prediction
```

### 3. Open the Notebook

Open:

```text
Fake_News_Prediction.ipynb
```

using either:

- Jupyter Notebook
- JupyterLab
- Google Colab

### 4. Install Required Libraries

```bash
pip install numpy pandas nltk scikit-learn
```

### 5. Download NLTK Stopwords

Run:

```python
import nltk
nltk.download('stopwords')
```

### 6. Provide the Dataset

The current notebook expects the dataset at:

```text
/content/train.csv
```

If you are running the notebook locally, update the dataset path accordingly.

### 7. Run the Notebook

Execute the notebook cells sequentially to perform:

```text
Data Loading
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
Train/Test Split
    ↓
Model Training
    ↓
Evaluation
    ↓
Prediction
```

---

## ⚠️ Current Limitations

The current implementation has several limitations that can be addressed in future versions.

### 1. Feature Limitation

The current model uses:

```text
Author + Title
```

as the textual feature.

Although the dataset contains the complete article `text` column, it is not currently included in the model input.

### 2. TF-IDF Preprocessing Order

In the current notebook, TF-IDF is fitted before the train/test split.

A more rigorous machine-learning workflow would be:

```text
Raw Dataset
    ↓
Train/Test Split
    ↓
Fit TF-IDF only on Training Data
    ↓
Transform Training Data
    ↓
Transform Test Data
    ↓
Train Model
    ↓
Evaluate
```

This prevents information from the test set from influencing the feature extraction process.

### 3. Evaluation Metrics

The current notebook primarily evaluates the model using accuracy.

A more comprehensive evaluation should include:

- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC where appropriate

### 4. Model Selection

The current implementation uses Logistic Regression as the primary classifier.

Additional suitable classification models could be evaluated and compared.

### 5. Deployment

The current repository contains a notebook-based machine learning pipeline rather than a deployed web application or production API.

### 6. Real-World Fact Checking

The model performs classification based on patterns learned from the labeled dataset. It does not independently verify whether a claim is factually true using external sources.

Therefore, the project should be considered an **NLP classification experiment**, not a definitive real-world fact-checking system.

---

## 🚀 Future Improvements

Possible improvements include:

### Data and Features

- Incorporate the full article text.
- Combine title, author and article body where appropriate.
- Perform additional data-quality checks.
- Investigate class distribution.

### Machine Learning

- Compare Logistic Regression with other suitable classifiers.
- Perform hyperparameter tuning.
- Use cross-validation.
- Analyze feature importance.
- Evaluate precision, recall and F1-score.

### NLP

- Experiment with different tokenization approaches.
- Compare stemming with lemmatization.
- Explore word embeddings.
- Explore transformer-based NLP models.

### Evaluation

- Generate a confusion matrix.
- Analyze false positives and false negatives.
- Evaluate the model on an independent dataset.
- Perform more robust validation.

### Deployment

A future version could provide a user interface where a user enters news content and receives a model prediction.

A possible architecture would be:

```text
User
  ↓
Web Interface
  ↓
Backend API
  ↓
Preprocessing Pipeline
  ↓
TF-IDF Vectorizer
  ↓
Trained Model
  ↓
Prediction
  ↓
Frontend Result
```

---

## 🎓 Key Learnings

Through this project, I gained practical experience in:

- Natural Language Processing
- Text preprocessing
- Missing-value handling
- Feature engineering
- TF-IDF vectorization
- Supervised machine learning
- Binary classification
- Logistic Regression
- Train/test splitting
- Model evaluation
- Python data analysis
- Building an end-to-end machine learning pipeline

The project helped me understand how **raw textual data can be transformed into numerical features and then used by a machine learning model to perform classification**.

---

## 💡 Key Takeaway

The main learning from this project was understanding the complete NLP machine-learning workflow:

```text
Raw Text
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
Feature Extraction
   ↓
Machine Learning
   ↓
Prediction
   ↓
Evaluation
```

This project provided practical exposure to the complete process of building a **text classification pipeline from raw data to model prediction**.

---

## 👩‍💻 Author

**Lakavath Tejaswini**

B.Tech – Metallurgy and Materials Engineering  
IIEST Shibpur

GitHub: [Teja234-hub](https://github.com/Teja234-hub)

---

## 📌 Disclaimer

This project is developed for educational and machine-learning experimentation purposes. The predictions generated by the model should not be treated as definitive verification of the factual accuracy of real-world news.
