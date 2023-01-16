# Disaster Response Pipeline Project

## Introduction
The aim of this project was to be able to dispher important disaster response messages more efficiently using machine learning algorithms. To do this, over 30,000 real-life messages were used, that had been encoded with 36 different disaster categories. Natural language processing and machine learning pipelines form a key element of the end product - being able to categorise different needs from the message input.

## Files in the repository


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
# DisasterResponsePipeline
