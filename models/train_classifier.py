# import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
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
from sklearn.externals import joblib

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

# define url identifier
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# setup stop words to filter
stop_words = set(stopwords.words('english'))


def load_data(database_filepath):

    '''
    INPUT: string directory of db
    OUTPUT: x message pd column, y categorical column labels, categorical names
    '''

    # read in file
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages', engine)
    x = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns

    return x, y, category_names


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

    '''
    INPUT: None
    OUTPUT: model pipeline object
    '''

    # text processing and model pipeline
    pipeline = pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=1,
                                                                 max_depth=50)))
        ])
    # define parameters for GridSearchCV
    parameters = {
    'clf__estimator__n_estimators': [10, 30, 50, 100],
    'clf__estimator__min_samples_leaf': [0.1, 0.5, 1],
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    INPUT: model object, test and validation test columns, unique categories list
    OUTPUT: print confusion_matrix for respective categories
    '''

    y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(column)
        print(classification_report(Y_test[column], y_pred[:, i]))

def save_model(model, model_filepath):
    # Export model as a pickle file
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
