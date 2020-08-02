# import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

# define url identifier
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# setup stop words to filter
stop_words = set(stopwords.words('english'))


def load_data(data_file):
    # read in file
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM Response', engine)
    x = df['message']
    y = df.iloc[:, 4:]
    return x, y, y.columns


def tokenize(text):

    '''
    INPUT: String to tokenise, detect and replace URLs
    OUTPUT: List of tokenised string items
    '''

    # Remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)

    text = [w for w in text.split() if not w in stop_words]

    # Join list to string
    text = " ".join(text)

    # Replace URLs if any
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Setup tokens and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Create tokens and lemmatize
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # text processing and model pipeline
    pipeline = pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    # define parameters for GridSearchCV
    parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)]
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline


def train(X, Y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    # fit model
    model.fit(X_train, y_train)

    # predict on test data
    y_pred = pipeline.predict(X_test)

    # output model test results
    labels = np.unique(y_pred)
    accuracy = (y_pred == y_test).mean()

    print('Labels:', labels)
    print('Accuracy:', accuracy)
    print('\nMean accuracy:', accuracy.mean())

    return model

def evaluate_model(model, X_test, Y_test, category_names):

    # print confusion_matrix for respective categories
    y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(column)
        print(classification_report(Y_test[column], y_pred[:, i]))

def export_model(model, model_filepath):
    # Export model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def run_pipeline(data_file):

    print('Loading data and ETL pipeline...')
    X, y = load_data(data_file)  # run ETL pipeline

    print('Building model...')
    model = build_model()  # build model pipeline

    print('Training model...')
    model = train(X, y, model)  # train model pipeline

    print('Exporting model to: ' + str(model_filepath) + '...')
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline