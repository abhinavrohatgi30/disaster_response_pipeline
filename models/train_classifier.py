# import libraries
import nltk

nltk.download(['punkt', 'wordnet'])

import pandas as pd
from sqlalchemy.engine import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import joblib
from custom_transformer import StartingVerbExtractor, tokenizer
import sys


def load_data(database_filepath):
    """
    Loads the data from the filesystem
    :param database_filepath: The path on the file system where the processed data is stored
    :return: the feature variables and target variables to be used to train and test the model
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', con=engine)
    colnames = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
                'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings',
                'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
                'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    return df['message'], df[colnames], colnames


def tokenize(text):
    return tokenizer(text)


def build_model():
    # Defines the pipeline to be used to build the model

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.75, max_features=5000)),
                ('tfidf', TfidfTransformer(use_idf=True))
            ])),

            ('verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, min_samples_split=2)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the accuracy, precision, recall and f1 score for every category individually
    :param model: the model to be evaluated
    :param X_test: The test features to be evaluated
    :param Y_test: The test output to be evaluated against
    :param category_names: the name of all the categories for which the metrics need to be generated
    :return: None
    """

    y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(y_pred, columns=category_names)
    for name in category_names:
        print(name)
        print(classification_report(Y_test[name], Y_pred[name]))


def save_model(model, model_filepath):
    """
     Saves the model onto the filesystem in a serialized form
    :param model: The model to be saved
    :param model_filepath: the path on the file system
    :return: None
    """
    joblib.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
