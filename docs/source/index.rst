.. Application of big data documentation master file, created by
   sphinx-quickstart on Tue Nov 16 16:49:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Application of big data's documentation!
===================================================

**Application of bigdata** (/Our project/) is a python project, that train us to 
apply tools and concepts seen in course.  It pulls data from the 
`DataSet ofHome Credit Risk Classification <https://www.kaggle.com/c/home-credit-default-risk/overview>`.

Part 1 :
---------

In the first part, we build a machine learning project using jupyter notebook, github, a conda envirenement and sphinx.
We tried to separate the different workflow into different scripts, one for the data preparation,
one for the data preparation, one for the feature engineering, one for the models training and a 
last one for the prediction.

**Data preparation** :

We first clean the dataset from all the NAN values and the columns that contained more than 30% of NAN values.

- **init()**, will return the cleaned dataset

**Feature engineering** :

We have done a correlation matrix, and from that we have kept the most correlacted features and deleted the least correlated ones.

- **matrice_corr(df_train,df_test)**, is a void function that show us the correlation matrix
- **setup_train(df_train,df_test)**, will return four values (X_train, X_test, y_train and y_test)

**Models training** : 

We had to train three models: XGboost, Random Forest and Gradient Boosting.
The XGboost model, is done with the optimized distributed gradient boosting library, XGboost.
The Ramdom Forest model, consists of many decision trees.
The Gradient Boosting model, is an ensemble of weak prediction models(decision trees).

- **XGBC_model(X_train,X_test,y_train,y_test,learning_rate,max_depth,scale_pos_weight)**
- **RF_model(X_train,X_test,y_train,y_test)**
- **GB_model(X_train,X_test,y_train,y_test)**

Those functions train the different models.

**Prediction** : 

All three model, succeed in predicting if a client could get a loan. Most had each around 0.91 of accuracy.

Part 2 :
---------

In this part, we got introduced to MLFLOW. We decided to track the parameters of the XGboost model.
It helped us to choose the best parameter, to have better result, with our model.

Part 3 :
---------

Finaly, we used SHAP Library on our XGboost model to understand it.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
