# Disaster Response Project

Udacity DataScientist Nanodegree

Andrzej Wodecki, 08.2019



## Project overview

The goal of this project is to:

1. analyze disaster massages data provided by [FigureEight](https://www.figure-eight.com/) 
2. create a model for classification of new incoming messages into a set of pre-defined categories
3. create a web app displaying key characteristics of data provided in a dataset and enabling an emergency worker to classify a new message.



## Project structure

There are 3 main components of the project: 

1. an ETL (Extract, Transform, Load) pipeline stored in a 'data' subfolder
2. a modelling component, where a preprocessed data is used to fit and evaluate a final model ('model' subfolder)
3. a web app, which display both data and a classification engine online ('webapp' subfolder).



### ETL pipeline

*data/process_data.py* file is used to:

1. load and merge the 'messages' and 'categories' datasets
2. perform necessary cleaning and transformations
3. store the resulting dataframe in a *SQLlite* database file



### Modelling component

*model/train_classifier.py* file is a real heart of the solution. The machine learning pipeline implemented there:

1. Loads data from a database
2. Splits the data into training and test datasets
3. Fits the model (applying GridSearchCV)
4. Evaluates the final model
5. Exports it as a pickle file.



### Web application

This final component uses Flask to generate a website enabling an emergency worker to classify a new message. It is stored in a *webapp* subfolder and consists of:

1. *run.py* app performing necessary data operations, generating figures and rendering a final website
2. two templates stored in *templates* subfolder: *master.html* with a main page and it's extension (*go.html*) displaying new message classification results.



# Implementation

To run the app:

1. Run the ETL pipeline:
   1. go to *data* folder
   2. type `python process_data.py disaster_messages.csv disaster_categories.csv disaster.db` to run *process_data.py*, read-in csv files and finally store them into *disaster.db* SQLlite file.
2. Run the ML (Machine Learning) pipeline:
   1. go to *model* folder
   2. type `python train_classifier.py ../data/disaster.db  model.pkl` to execute a ML pipeline, taking a *disaster.db* as input and storing a final model into a *model.pkl* file (pickle).
3. Finally, run the web app:
   1. go to *app* folder
   2. run `python run.py` and follow the on-screen instruction (just open http://0.0.0.0:3001 in Your browser).



# Requirements

You will need:

Flask==1.0.2
nltk==3.4
numpy==1.15.4
pandas==0.22.0
plotly==3.4.2
scikit-learn==0.20.1
SQLAlchemy==1.2.14



# Acknowledgments

1. Udacity.com: for a great idea for the project, and a 'starter' pack (useful scripts)
2. FigureEight.com for very good datasets.

