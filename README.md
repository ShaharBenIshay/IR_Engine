# IR_Engine

In this project we created an information retrival engine on all wikipedia files as part of our final project in "Information Retrival" course.
For a given query our engine tries to find the most relevant wikipedia documents for this given query.
In the Preproccess phase we used the help of the google storage cloud(GCP) and other python common libraries such as pandas, numpy and many more...

NOTE: all this project is written in Python and implemented using PyCharm, JupyterNoteBook and GoogleColab.

Project files:

* BMֹ_25_from_index: class object that implement an index based on BM 25 score.

* inverted_index_gcp: used to create an inverted index object. this is the base index object.

* search_fronted: Used to create the server-side using flask, receive queries from clients and provide answers.

* create_inverted_indexes_for_gcp: in this file we create 3 inverted indexes each based on a different part of the document(title, body and anchor). we used spark to     perform operations on all the documents. after the creation of each index we wrote it to a bucket in our GCP project.

* IR_Engine_Performance: in this file we splitted 30 queries to train and test. then we tested 5 different versions of our engine and measure the MAP@40 and the average retrival time. after the train phase we choose the best version and tested it on the test set. 

