# Data Science Demo

This repository was created to practice and demonstrate typical data science workflows on publicly available data.
It is not meant to be complete or perfect but rather as a sandbox.

The bike sharing data set used in this repo is taken from:

https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

Some of the main points I wanted to practice in this project include:

  1. Exploratory data analysis, including visualization.
  2. Utilizing sklearn pipelines, a powerful tool for data science tasks.
  3. Creating custom transformers using sklearn.
  
  
## Structure of this Project

There are currently two notebooks for this project:

  * ```1_EDA.ipynb```: exploratory data analysis, formulating some hypotheses and questions
  * ```2_modelfitting.ipynb```: model fitting process, based on the findings in the first notebook

Any additional code that is used in the second notebook can be found in the folder ```util/```. It contains the following files:

  * ```custom_transformers.py```: custom skleanr transformers that are used in the second notebook
  * ```ML_eval.py```: Some functions I wrote to evaluate machine learning  models conveniently.
  * ```prepare_data.py```: A function that summarizes all the preprocessing steps that are done in the EDA notebook. This function is currently not used in this project (as the second notebook uses pipelines to preprocess the data).
