import sys
import nltk
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine
import pickle 

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Function to load in the content of messages and categories table
    from SQLite database into variables X and y.
    """
    
    engine = create_engine('sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    df.dropna()
    X = df.message
    Y = df[df.columns[4:]] 

    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    """
    Function to tokenize the text, lemmatize it, and change text to lower case.
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
    This function defines a ML pipeline with best parameters found using GridSearch. 
    The model is fitted on the training data.
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]) 

    parameters = {'clf__estimator__n_estimators': [10, 50]}

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Function to evaluate the ML models performance.
    """
    
    y_pred = model.predict(X_test)

    print(classification_report(y_test.values, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """
    Function to save the trained model into a pickle file.
    """
    
    pickle.dump(model,  open('model.pkl', 'wb'))


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