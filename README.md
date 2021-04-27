# Disaster Response Pipeline Project

This project is aimed at training a classifier that helps in assigning various disaster responses into a set of categories 
that help in providing prompt and actionable response to these messages. 

The project consists of 3 different stages:

1. <b><u>Preparing the Training and Test Data</u></b> : The resources for this stage can be found under the `data` directory and consists of a `process_data.py`
file that loads a CSV file that contains the raw data around various disaster response messages and their classification.
The process_data.py script cleans the raw data and structures it to be in a processable format  and stores it into a local DB on the file system as provided
in the runtime arguments.

2. <b><u>Training the classifier</u></b> : The resources for this stage can be found under the `models` directory and consists of a `train_classifier.py`
file that uses the local DB created to read the processed data and use that to train as well as test a Multi output classifier using a ML pipeline that has the hyper-parameters
that have been optimised after doing cross validation on the training data. This stage also uses a `custom_transformer.py` file to make use of a custom transformer
in the ML pipeline which is used to detect the existence of a Verb in the disaster response message. The script after generating the model serializes it to a file on the file system
as provided in the runtime arguments.

3. <b><u>Running the classifier as a Flask Service</u></b> : The resources for this stage can be found under the `app` directory and consists of a `run.py`
, which is a flask application that loads the trained classifier and hosts it so that users can interact with it realtime and classify their response messages
into a set of categories. It also uses a few resources to generate a web app endpoint that are present under the `app/templates` directory.


### Instructions to run the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
