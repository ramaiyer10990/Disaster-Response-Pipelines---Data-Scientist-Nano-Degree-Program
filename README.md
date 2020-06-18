# Disaster-Response-Pipelines---Data-Scientist-Nano-Degree-Program

# Data Scientist Nanodegree

## Project: Disaster Response Pipeline

## Table of Contents

- [Project Overview](#overview)
- [Project Components](#components)
  - [ETL Pipeline](#etl_pipeline)
  - [Machine Learning Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
- [Running](#run)
  - [Data Cleaning](#cleaning)
  - [Training Classifier](#training)
  - [Starting the Web App](#starting)
- [Screenshots](#screenshots)
- [Software Requirements](#sw)
- [File Descriptions](#files)
- [Credits and Acknowledgements](#credits)

***

<a id='overview'></a>

## 1. Project Overview

This project is part of the Data Scientist Nanodegree Program by Udacity Nanodegree program that uses a <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> dataset. The initial dataset contains pre-labeled real messages from real-life disaster events. The aim of the project is to apply skills to analyze disaster data from from Figure Eight dataset to build a model for an API that classifies disaster messages that have occured in response to real life events.

'_data_' directory contains the Figure Eight data set which are real messages sent in respone to a disaster event that occured. I created a machine learning pipeline to categorize these events so that appropriate disaster relief agency is notified.

This project includes a Flask web app where an essential worker can input a new message and get classifited text in the form of results in several categories. The web app also displays data visualizations.

The project is divided into the following sections:

- Data Processing, using ETL Pipeline to extract data from the source, clean data and load it into a database
- Machine Learning Pipeline(ML Pipeline) for training a model to classify messages to categories
- A Flask Web App to show model results intuitively in real-time


<a id='components'></a>

## 2. Project Components

There are three components of this project:

<a id='etl_pipeline'></a>

### 2.1. ETL Pipeline

File _data/process_data.py_:

- Loads the `messages` and `categories` dataset
- Merges and cleans the data
- Stores data in a SQLite database i.e.DisasterResponse.db

<a id='ml_pipeline'></a>

### 2.2. Machine Learning Pipeline

File _models/train_classifier.py_:

- Loads data from SQLite database i.e.DisasterResponse.db
- Creates training and test datasets
- Builds and Trains on a text data processing and machine learning pipeline (Decision Tree classifier)
- Uses GridSearchCV to optimize parameters
- Score on test dataset
- Exports the cleaned data on a final model as a pickle file

<a id='flask'></a>

### 2.3. Flask Web App

<a id='eg'></a>
File _app/run.py_:

Run `python run.py` from app directory to start the web app where users can enter their messages, i.e., message sent during a disaster. App classifies messages into categories.



<a id='run'></a>

## 3. Running

There are three steps to get up and runnning with the web app if you want to start from ETL process.

<a id='cleaning'></a>

### 3.1. Data Cleaning

Run the following command to execute ETL pipeline that cleans and stores data in database:

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

The first two arguments - input data
third argument - Output SQLite Database name to store the cleaned data.



<a id='training'></a>

### 3.2. Training Classifier

Run the following command to execute ML pipeline that trains classifier and saves:

`python models/train_classifier.py data/DisasterResponse.db models/model_classifier.pkl`

Loads data from SQLite database to train the model and save the model to a pickle file.


<a id='starting'></a>

### 3.3. Starting the web app

Run the following command in the app's directory to run the web app.

`python run.py`

This will start the web app and will direct you to http://0.0.0.0:3001/ where you can enter messages and get classification results.


<a id='screenshots'></a>

## 4. Screen capture

<img src="images/Image 6-17-20 at 6.29 PM.png">

<img src="images/Image 6-17-20 at 6.29-1 PM.png">

<img src="images/Image 6-17-20 at 6.30 PM.png">

**_ Message provided as input to evaluate ML model performance_**

<img src="images/Image 6-17-20 at 6.33 PM.png">

**_Clicking Classify Message, will highlight the relevant message categories_**

<img src="images/Image 6-17-20 at 6.31 PM.png">

<img src="images/Image 6-17-20 at 6.32 PM.png">

<a id='sw'></a>

## 5. Software Requirements

- Python 3.7
- Machine Learning Libraries: NumPy, Pandas, Sciki-Learn or sklearn, pickle
- Natural Language Process  libraries: NLTK, re
- SQLlite Database Libraries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly,
- python related: sys, warnings

## File Descriptions <a name="files"></a>

There are three main folders:
1. data
- disaster_categories.csv: dataset contains categories
- disaster_messages.csv: dataset contains messages
- process_data.py: ETL pipeline scripts to load, clean, merge and store data into a database
- DisasterResponse.db: SQLite database containing processed messages and categories data
2. models
- train_classifier.py: machine learning pipeline scripts to train, and save a model
- classifier.pkl: saved the model as .pkl(pickle) file.
3. app
- run.py: Python script integrates all above files and runs the Flask web app.
- templates that contain html file for the web application

<a id='credits'></a>

## 6. Credits and Acknowledgements

* [Figure Eight](https://www.figure-eight.com/) for providing the dataset implemented in the project.
* [Udacity](https://www.udacity.com/) for a highly intense project that honed my skills.
