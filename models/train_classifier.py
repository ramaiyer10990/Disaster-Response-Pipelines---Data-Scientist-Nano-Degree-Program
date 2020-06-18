import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from sqlalchemy import create_engine
import sys
import re
import pickle
from sklearn.pipeline import Pipeline,FeatureUnion
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, make_scorer, accuracy_score, f1_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    table_name = 'Disaster_messages'
    engine = create_engine("sqlite:///DisasterResponse.db")
    df = pd.read_sql_table(table_name,engine)
    X = df["message"]
    Y = df.drop(["genre","message","id","original"], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    tokenize and transform input text. Output is to return cleaned text
    Inputs:
    list of the messages
    
    Returns: A list of words into numbers/root form of the messages words
    """
    #Normalizing text by converting everything to lower case:
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize 'text'
    tokenizedwords = word_tokenize(text)
    
    #Normalization of word tokens and removal of stop words
    normalizertokens = PorterStemmer()
    stop_words = stopwords.words("english")
    
    normalizedwords = [normalizertokens.stem(word) for word in tokenizedwords if word not in stop_words]
    
    return normalizedwords

def build_model():
    """
    Build a model pipeline for classifying the disaster messages
    
    Output is a tuned classified model for processing text messages
    """
    modelpipeline = pipeline_rfc = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

    #Parameter grid
    parameters = {'clf__estimator__max_depth': [10, 50, None],
                  'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv =  GridSearchCV(pipeline_rfc, parameters)
    
    # create model
    model = GridSearchCV(estimator=modelpipeline,
            param_grid=parameters,
            verbose=3,
            #n_jobs = -1,
            cv=2)

    return model

def get_results(Y_test, Y_pred):
    """
    Display the results and model them into a dataframe format
    
    Arguments
    Input: Y_test -> test labels
    Returns: Y_pred -> predicted labels
    """
    modelresults = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for ctg in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[ctg], Y_pred[:,num], average='weighted')
        modelresults.at[num+1, 'Category'] = ctg 
        modelresults.at[num+1, 'f_score'] = f_score
        modelresults.at[num+1, 'precision'] = precision
        modelresults.at[num+1, 'recall'] = recall  
        num +=1
    print('Aggregated f_score:', modelresults['f_score'].mean())
    print('Aggregated precision:', modelresults['precision'].mean())
    print('Aggregated recall:', modelresults['recall'].mean())
    print('Accuracy:', np.mean(Y_test.values == Y_pred))
    return modelresults
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    model -> Scikit ML Pipeline
        X_test -> test message features
        Y_test -> test labels
        category_names -> label names
    """
    #Get results and adding them to dataframe
    Y_pred = model.predict(X_test)
    modelresults = get_results(Y_test, Y_pred)
    print(modelresults)


def save_model(model, model_filepath):
    """
    Function: Save a pickle file of the classified model
    Arguments:
    model: GridSearchCV or Pipeline object
    model_filepath (str): the destination path of pickle (.pkl) file
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