# Sentiment-Analysis-on-Twitter-Data

## Project Overview
This project focuses on performing sentiment analysis on Twitter data using machine learning and deep learning models. The sentiment classes in the dataset are:

1. Positive (1)
2. Negative (2)
3. Neutral (0)

#### Dataset
The dataset consists of 3,534 rows with multiple columns such as:

Text: The tweet content
Sentiment: Target labels (0: Neutral, 1: Positive, 2: Negative)
Other columns like user details and geographic information are not used for model building but are part of the dataset.

## Preprocessing
### Data cleaning steps included:

Removing hashtags, URLs, mentions, and special characters from the tweets.
Lowercasing all text for consistency.
### Feature Engineering
TF-IDF vectorization was applied to convert the text data into numerical form for model training.

### Models Used
Multiple machine learning and deep learning models were trained to predict sentiment, including:

Logistic Regression
Random Forest
Support Vector Machine (SVM)
AdaBoost
Transformers (BERT)
Data Augmentation

To improve model performance, various data augmentation techniques such as back translation and random word insertion were applied to the training set.

### Evaluation
Model performance was evaluated using accuracy, confusion matrix, precision, recall, F1-score, and AUC.

### Results
Despite rigorous preprocessing and data augmentation, the models initially achieved accuracies in the range of 62-64%, with further improvements observed after applying back translation and hyperparameter tuning.

## Technologies Used
Python
Scikit-learn for machine learning models
Transformers (Hugging Face) for BERT-based models
TF-IDF Vectorization for text data
Data Augmentation to enhance training data
GridSearchCV for hyperparameter tuning

### How to Run
#### Clone the repository.
Install the required packages:

pip install -r requirements.txt

Run the notebook to preprocess data, train models, and evaluate performance.
