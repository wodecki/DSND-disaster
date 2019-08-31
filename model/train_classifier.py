import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from statistics import mean

from joblib import dump

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from cleaned messages dataframe

    Args:
    database_filepath: a filepath to a database with categorized and cleaned disaster messages

    Returns:
    X: a numpy array with message texts
    y: a dataframe with corresponding categories
    category_names: a list of category names
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisastersData', con = conn)

    category_names = df.iloc[:, 4:].columns
    X = df.message.values
    y = df[category_names]

    return X, y, category_names


def tokenize(text):
    """
    Tokenize a given text

    Args:
    text: a text to be tokenized

    Returns:
    clean_tokens: a list of cleaned (tokenized) words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build and train a model using RandomForestClassifier algorithm

    Args: None

    Returns:
    model: a trained RandomForestClassifier model  
    """
    # define a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    params = {
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__n_estimators': [10, 20]
    }

    model = GridSearchCV(pipeline, param_grid=params, verbose = 2, n_jobs=-1) #verbose = 2 to report job details, n_jobs = -1 to use all processors
   
    return model

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate Your model

    Args: 
    model: model to evaluate
    X_test: a dataframe with test messages
    Y_test: a dataframe with corresponding categories
    category_names: a list with categories

    Returns:
    printout: a classification_report data (the weighted averages of f1 score, precision and 
    recall for each output category of the dataset)
    printout: test accuracies for each catogory
    """

    predictions = model.predict(X_test)
    print(classification_report(predictions, y_test.values, target_names=category_names))
    
    accuracies = []

    for i in range(35):
        category = y_test.columns[i]
        accuracy = accuracy_score(y_test.values[:,i], predictions[:,i])
        accuracies.append(accuracy)
        print('Accuracy score for {} = {}'.format(category, accuracy))
    
    print('A mean accuracy score for this model = {}'.format(mean(accuracies)))
    

def save_model(model, model_filepath):
    """
    Save a trained model to a pickle file

    Args: 
    model: a model to save
    model_filepath: a model filepath
    
    Returns: a model file 
    """
    pickle.dump(model, open(model_filepath, 'wb'))

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